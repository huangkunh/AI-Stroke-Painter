#!/usr/bin/env python3
"""
Image -> stroke-action sequence.

This is the entry point of the model side of AI-Stroke-Painter. Given a
target image, it produces a sequence of stroke actions in the canonical
schema consumed by `converter/transform.py`:

    [x_start, y_start, x_end, y_end,
     color_r, color_g, color_b, color_a, brush_radius]

Two backends are available:

  --mode rl    (default when torch + weights are available)
      Loads the pretrained Agent from Learning-to-Paint and runs the
      recurrent policy for `--max-steps` strokes. The neural renderer
      simulates the canvas so the agent can observe its own progress.

  --mode lite  (always available; pure numpy / opencv)
      A deterministic heuristic painter that:
        1. colour-quantises the image into a small palette,
        2. lays down broad colour-block strokes (large radius, low alpha),
        3. adds mid-frequency strokes guided by region edges,
        4. finishes with thin detail strokes along strong gradients.
      This produces visually pleasing, human-like stroke orderings without
      any neural network and is the recommended mode for a quick demo.

Output: a JSON file `raw_strokes.json` containing a list of action dicts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import List, Dict

import cv2
import numpy as np
from PIL import Image

# Make sibling module importable when run as `python model/inference.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

# Minimum acceptable size for the pretrained weights file (500 KB).
# Real Learning-to-Paint actor weights are several MB; anything smaller is
# almost certainly a truncated / error-page download.
MIN_WEIGHTS_BYTES = 500_000

# Optional expected SHA256 of actor_final.pth. When the upstream release
# rotates the file, set this to the new hash to enforce integrity. Leave as
# None to skip hash verification (size check still applies).
EXPECTED_SHA256 = None


def load_image(path: str, size: int = 512) -> np.ndarray:
    """Load an image and resize it to (size, size, 3) float32 in [0, 1]."""
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def action_dict(x0, y0, x1, y1, r, g, b, a, radius) -> Dict:
    """Canonical action schema (all coordinates normalised to 0..1)."""
    return {
        "x_start": float(x0),
        "y_start": float(y0),
        "x_end": float(x1),
        "y_end": float(y1),
        "color_r": float(r),
        "color_g": float(g),
        "color_b": float(b),
        "color_a": float(a),
        "brush_radius": float(radius),
    }


def verify_weights(weights_path: str) -> bool:
    """Verify a pretrained weights file is intact.

    Checks (in order):
      1. File exists.
      2. File size >= MIN_WEIGHTS_BYTES.
      3. (Optional) SHA256 matches EXPECTED_SHA256 when that constant is set.

    Returns True if the file passes all checks, False otherwise. Prints a
    descriptive WARNING line for each failed check so the caller can report
    the reason to the user.
    """
    if not os.path.isfile(weights_path):
        print(f"[inference][rl] WARNING: weights file not found: {weights_path}")
        return False

    size = os.path.getsize(weights_path)
    if size < MIN_WEIGHTS_BYTES:
        print(f"[inference][rl] WARNING: weights file too small "
              f"({size} bytes < {MIN_WEIGHTS_BYTES} expected).")
        print(f"[inference][rl] WARNING: the download may be truncated or "
              f"an error page. Re-run `bash model/download_weights.sh`.")
        return False

    if EXPECTED_SHA256 is not None:
        h = hashlib.sha256()
        with open(weights_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        actual = h.hexdigest()
        if actual.lower() != EXPECTED_SHA256.lower():
            print(f"[inference][rl] WARNING: SHA256 mismatch.")
            print(f"[inference][rl] WARNING:   expected: {EXPECTED_SHA256}")
            print(f"[inference][rl] WARNING:   actual:   {actual}")
            print(f"[inference][rl] WARNING: the file may be corrupted or "
                  f"replaced. Re-download or update EXPECTED_SHA256.")
            return False

    return True


# ---------------------------------------------------------------------------
# RL backend (Learning-to-Paint)
# ---------------------------------------------------------------------------

def run_rl_inference(image: np.ndarray, weights_path: str, max_steps: int,
                     device: str = "cpu") -> List[Dict]:
    """Run the pretrained Agent for `max_steps` strokes.

    Falls back to the lite painter if torch or the weights are unavailable.
    All failure paths print a clear ``[inference][rl] WARNING`` line so the
    user always knows why RL mode was skipped.
    """
    # --- 1. torch + network availability --------------------------------
    try:
        import torch
        from network import build_agent, build_renderer, decode_action, ACTION_DIM
    except Exception as e:  # pragma: no cover
        print(f"[inference][rl] WARNING: torch/network unavailable ({e}).")
        print(f"[inference][rl] WARNING: falling back to lite mode.")
        return run_lite_inference(image, max_steps=max_steps)

    # --- 2. weights file existence + integrity --------------------------
    if not verify_weights(weights_path):
        print(f"[inference][rl] WARNING: run `bash model/download_weights.sh` first,")
        print(f"[inference][rl] WARNING: or use --mode lite for a quick demo.")
        print(f"[inference][rl] WARNING: falling back to lite mode.")
        return run_lite_inference(image, max_steps=max_steps)

    # --- 3. model construction + weight loading -------------------------
    device_t = torch.device(device)
    try:
        agent = build_agent().to(device_t).eval()
        renderer = build_renderer().to(device_t).eval()

        sd = torch.load(weights_path, map_location=device_t)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        agent.load_state_dict(sd, strict=False)
        print(f"[inference][rl] loaded agent weights from {weights_path}")
    except Exception as e:
        print(f"[inference][rl] WARNING: failed to load weights ({e}).")
        print(f"[inference][rl] WARNING: the file may be corrupted or incompatible.")
        print(f"[inference][rl] WARNING: falling back to lite mode.")
        return run_lite_inference(image, max_steps=max_steps)

    # --- 4. inference loop ----------------------------------------------
    target = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device_t)
    canvas = torch.zeros_like(target)
    h = agent.init_hidden(1, device_t)

    actions: List[Dict] = []
    with torch.no_grad():
        for step in range(max_steps):
            action, h = agent(target, canvas, h)
            actions.append(decode_action(action[0]))
            # advance the neural canvas so the agent sees its own strokes
            canvas = renderer(canvas, action)
            if (step + 1) % 25 == 0:
                print(f"[inference][rl] step {step + 1}/{max_steps}")
    return actions


# ---------------------------------------------------------------------------
# Lite backend (heuristic painter, no torch required)
# ---------------------------------------------------------------------------

def _quantize(img: np.ndarray, k: int = 8) -> tuple:
    """K-means colour quantisation. Returns (quantised_img, palette)."""
    h, w, _ = img.shape
    pts = img.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.05)
    compactness, labels, centers = cv2.kmeans(pts, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    quant = centers[labels.flatten()].reshape(h, w, 3)
    return quant, centers


def _sample_strokes_in_region(mask: np.ndarray, target: np.ndarray,
                              n_strokes: int, radius: float, alpha: float,
                              rng: np.random.Generator) -> List[Dict]:
    """Sample long-ish strokes inside a binary region mask.

    Each stroke starts at a random foreground pixel and walks along the
    region's principal direction (estimated from the local gradient) for
    a few steps. This mimics how a human fills a colour block with a few
    sweeping brush motions.
    """
    if n_strokes <= 0:
        return []
    ys, xs = np.where(mask > 0)
    if len(xs) < 4:
        return []
    h, w = mask.shape
    actions: List[Dict] = []
    # Pre-compute a distance-transform so we prefer stroke centres that are
    # far from the region boundary (avoids colour bleeding).
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    probs = dist.flatten().astype(np.float64) ** 2
    total = probs.sum()
    if total < 1e-9:
        # All-zero distance transform (degenerate mask); fall back to uniform
        probs = np.ones(len(probs)) / len(probs)
    else:
        probs /= total

    # Ensure we don't request more samples than there are non-zero entries
    n_nonzero = int((probs > 0).sum())
    n_pick = min(n_strokes, n_nonzero)
    if n_pick <= 0:
        return []
    idxs = rng.choice(len(probs), size=n_pick, replace=False, p=probs)
    for flat_idx in idxs:
        sy, sx = divmod(int(flat_idx), w)
        # local colour
        r, g, b = target[sy, sx]
        # random direction, length scaled by region "thickness"
        length = float(rng.uniform(0.04, 0.12))
        angle = float(rng.uniform(0, 2 * np.pi))
        dx, dy = length * np.cos(angle), length * np.sin(angle)
        ex, ey = sx / w + dx, sy / h + dy
        actions.append(action_dict(
            sx / w, sy / h, ex, ey, r, g, b, alpha, radius
        ))
    return actions


def _edge_strokes(target: np.ndarray, edges: np.ndarray, n_strokes: int,
                  radius: float, alpha: float, rng: np.random.Generator) -> List[Dict]:
    """Short strokes along strong edges (detail pass)."""
    if n_strokes <= 0:
        return []
    ys, xs = np.where(edges > 0)
    if len(xs) < 4:
        return []
    h, w = edges.shape
    # Sample the strongest edges preferentially.
    weights = edges[ys, xs].astype(np.float64) ** 2
    total = weights.sum()
    if total < 1e-9:
        weights = np.ones(len(weights)) / len(weights)
    else:
        weights /= total
    # Ensure we don't request more samples than available
    n_pick = min(n_strokes, len(xs))
    if n_pick <= 0:
        return []
    pick = rng.choice(len(xs), size=n_pick, replace=False, p=weights)
    actions: List[Dict] = []
    for i in pick:
        sx, sy = int(xs[i]), int(ys[i])
        r, g, b = target[sy, sx]
        # Walk a short distance along the edge tangent.
        gy, gx = np.gradient(target[:, :, 0].astype(np.float32))
        tx, ty = -gy[sy, sx], gx[sy, sx]
        norm = (tx * tx + ty * ty) ** 0.5 + 1e-6
        tx, ty = tx / norm, ty / norm
        length = float(rng.uniform(0.015, 0.04))
        ex = sx / w + tx * length
        ey = sy / h + ty * length
        actions.append(action_dict(
            sx / w, sy / h, ex, ey, r, g, b, alpha, radius
        ))
    return actions


# ---------------------------------------------------------------------------
# Image-type classification (photo / sketch / illustration)
# ---------------------------------------------------------------------------

def classify_image(image: np.ndarray) -> tuple:
    """Classify an image as 'photo', 'sketch', or 'illustration'.

    Uses simple, fast heuristics based on colour saturation and edge density:
      - sketch:       low saturation (mostly grey/white) + high edge density
      - illustration: high saturation + few unique colours (flat regions)
      - photo:        everything else (natural colour distribution)

    Returns (type_name, strategy_dict) where strategy_dict tunes the lite
    painter's pass budgets and stroke parameters for the detected type.
    """
    h, w, _ = image.shape
    # Work on a small downsampled version for speed
    scale = 64.0 / max(h, w)
    small = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)

    # Saturation: convert to HSV, measure mean S
    hsv = cv2.cvtColor((small * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    mean_sat = float(hsv[:, :, 1].mean()) / 255.0
    mean_val = float(hsv[:, :, 2].mean()) / 255.0

    # Edge density: Canny on grayscale
    gray = cv2.cvtColor((small * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(edges.sum() / 255.0) / (gray.shape[0] * gray.shape[1])

    # Colour uniqueness: count distinct quantised colours
    quantised = (small * 15).astype(np.uint8)  # 16 levels per channel
    flat = quantised.reshape(-1, 3)
    unique_colours = len(set(map(tuple, flat[::7])))  # sample every 7th pixel

    # Classification rules
    if mean_sat < 0.15 and edge_density > 0.08:
        img_type = "sketch"
    elif mean_sat > 0.35 and unique_colours < 80:
        img_type = "illustration"
    else:
        img_type = "photo"

    # Strategy per type
    strategies = {
        "photo": {
            "name": "balanced",
            "block_frac": 0.45, "block_radius": 14.0, "block_alpha": 0.55,
            "mid_frac": 0.30,   "mid_radius": 5.0,   "mid_alpha": 0.75,
            "fine_radius": 1.8,  "fine_alpha": 0.90,
        },
        "sketch": {
            "name": "edge-focused",
            "block_frac": 0.0,   # no colour blocking for sketches
            "block_radius": 8.0, "block_alpha": 0.4,
            "mid_frac": 0.45,    "mid_radius": 3.0,   "mid_alpha": 0.85,
            "fine_radius": 1.2,  "fine_alpha": 0.95,
        },
        "illustration": {
            "name": "colour-focused",
            "block_frac": 0.60,  "block_radius": 16.0, "block_alpha": 0.60,
            "mid_frac": 0.25,    "mid_radius": 6.0,   "mid_alpha": 0.70,
            "fine_radius": 2.5,  "fine_alpha": 0.85,
        },
    }
    return img_type, strategies[img_type]


def run_lite_inference(image: np.ndarray, max_steps: int = 600) -> List[Dict]:
    """Heuristic, deterministic painter. See module docstring.

    Automatically detects the image type (photo / sketch / illustration) and
    adjusts the stroke strategy accordingly:
      - photo:        balanced 3-pass strategy (block + mid + fine)
      - sketch:       skip colour blocking, emphasise fine edge strokes
      - illustration: stronger colour blocking, fewer fine details
    """
    rng = np.random.default_rng(42)
    h, w, _ = image.shape

    # 0) image-type detection -> strategy tuning
    img_type, strategy = classify_image(image)
    print(f"[inference][lite] image type: {img_type} -> strategy: {strategy['name']}")

    # 1) colour quantisation
    quant, palette = _quantize(image, k=8)

    # 2) per-colour masks (sorted by area, largest first -> background first)
    masks = []
    for c in palette:
        diff = np.abs(quant - c).sum(axis=2)
        mask = (diff < 0.05).astype(np.uint8) * 255
        if mask.sum() < 200:
            continue
        # clean up
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        masks.append((c, mask))
    masks.sort(key=lambda cm: -cm[1].sum())

    actions: List[Dict] = []

    # 3) PASS 1 - broad colour blocking (large radius, low alpha)
    #    Skipped entirely for sketches (they have little colour to block).
    block_budget = max(0, int(max_steps * strategy["block_frac"]))
    if block_budget > 0 and masks:
        per_region = max(8, block_budget // max(1, len(masks)))
        for color, mask in masks:
            r, g, b = color
            actions.extend(_sample_strokes_in_region(
                mask, image, per_region,
                radius=strategy["block_radius"],
                alpha=strategy["block_alpha"], rng=rng))
    print(f"[inference][lite] pass 1 (colour block): {len(actions)} strokes")

    # 4) PASS 2 - mid-frequency strokes guided by region edges
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges_mid = cv2.Canny(gray, 60, 140)
    edges_mid = cv2.dilate(edges_mid, np.ones((2, 2), np.uint8))
    mid_budget = max(0, int(max_steps * strategy["mid_frac"]))
    actions.extend(_edge_strokes(image, edges_mid, mid_budget,
                                 radius=strategy["mid_radius"],
                                 alpha=strategy["mid_alpha"], rng=rng))
    print(f"[inference][lite] pass 2 (mid edges):   {len(actions)} strokes total")

    # 5) PASS 3 - fine detail along strong gradients
    edges_fine = cv2.Canny(gray, 120, 240)
    fine_budget = max(0, max_steps - len(actions))
    actions.extend(_edge_strokes(image, edges_fine, fine_budget,
                                 radius=strategy["fine_radius"],
                                 alpha=strategy["fine_alpha"], rng=rng))
    print(f"[inference][lite] pass 3 (fine detail): {len(actions)} strokes total")

    return actions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Image -> stroke actions.")
    ap.add_argument("--image", required=True, help="Path to the target image.")
    ap.add_argument("--out", default="raw_strokes.json",
                    help="Output JSON path (default: raw_strokes.json).")
    ap.add_argument("--mode", choices=["auto", "rl", "lite", "hierarchical", "attention"], default="auto",
                    help="Inference backend. 'auto' picks rl if torch + weights "
                         "are available, otherwise lite. 'hierarchical' uses the "
                         "4-layer painting strategy. 'attention' uses the 5-layer "
                         "attention-based strategy with style support.")
    ap.add_argument("--style", default="default",
                    help="Painting style for attention mode (default/oil/watercolor/sketch/anime).")
    ap.add_argument("--max-steps", type=int, default=600,
                    help="Maximum number of strokes to emit.")
    ap.add_argument("--size", type=int, default=512,
                    help="Internal canvas size (image is resized to this).")
    ap.add_argument("--weights", default=os.path.join(os.path.dirname(__file__),
                                                     "pretrained", "actor_final.pth"),
                    help="Path to the pretrained Agent weights (rl mode only).")
    ap.add_argument("--device", default="cpu", help="torch device (rl mode only).")
    args = ap.parse_args()

    print(f"[inference] loading image: {args.image}")
    image = load_image(args.image, size=args.size)
    print(f"[inference] image ready: {image.shape}")

    mode = args.mode
    if mode == "auto":
        try:
            import torch  # noqa: F401
            if os.path.isfile(args.weights):
                mode = "rl"
            else:
                print(f"[inference] auto: weights not found at {args.weights}, using lite mode.")
                mode = "lite"
        except Exception:
            print("[inference] auto: torch not installed, using lite mode.")
            mode = "lite"
    print(f"[inference] backend: {mode}")

    # Run inference. If RL mode fails at ANY point (weights, model build,
    # inference loop), fall back to lite mode so the pipeline never breaks.
    actions: List[Dict]
    if mode == "rl":
        try:
            actions = run_rl_inference(image, args.weights, args.max_steps, args.device)
        except Exception as e:
            print(f"[inference] WARNING: RL inference raised an unexpected error ({e}).")
            print(f"[inference] WARNING: falling back to lite mode to keep the pipeline running.")
            actions = run_lite_inference(image, max_steps=args.max_steps)
    elif mode == "hierarchical":
        try:
            from hierarchical_painter import build_hierarchical_painter
            painter = build_hierarchical_painter(canvas_size=args.size)
            actions = painter.paint(image, max_strokes=args.max_steps, use_neural=False)
        except Exception as e:
            print(f"[inference] WARNING: hierarchical inference failed ({e}).")
            print(f"[inference] WARNING: falling back to lite mode.")
            actions = run_lite_inference(image, max_steps=args.max_steps)
    elif mode == "attention":
        try:
            from attention_painter import build_attention_painter
            painter = build_attention_painter(canvas_size=args.size, style=args.style)
            actions = painter.paint(image, max_strokes=args.max_steps, use_attention=True)
        except Exception as e:
            print(f"[inference] WARNING: attention inference failed ({e}).")
            print(f"[inference] WARNING: falling back to lite mode.")
            actions = run_lite_inference(image, max_steps=args.max_steps)
    else:
        actions = run_lite_inference(image, max_steps=args.max_steps)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(actions, f, ensure_ascii=False)
    print(f"[inference] wrote {len(actions)} strokes -> {args.out}")


if __name__ == "__main__":
    main()
