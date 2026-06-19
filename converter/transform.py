#!/usr/bin/env python3
"""
Convert raw model actions into the rendering-engine JSON format.

Input  : raw_strokes.json  (list of action dicts produced by model/inference.py)
Output : output_strokes.json (flat instruction stream for renderer/engine.js)

The engine consumes a flat array of instructions. Each instruction is a
2-tuple `[op, payload]` where `op` is one of:

    ["background", "#RRGGBB"]            set background colour
    ["colour",    "#RRGGBB"]             set brush colour
    ["width",     <number>]              set base brush width (px)
    ["alpha",     <0..1>]                set global alpha
    ["line",      <brushId>, x1,y1,p1, x2,y2,p2, ...]   draw a stroke

Brush selection (per the project spec):
    brush_radius <  3   ->  brushId = 5  (压感v3, pressure-sensitive, fine line)
    brush_radius >= 3   ->  brushId = 0  (马克笔, marker, flat colour block)

Pressure curve:
    Each model stroke [x0,y0,x1,y1] is linearly interpolated into N points
    (N = max(5, ceil(stroke_length_px / step))). A bell-shaped pressure
    profile is applied so the stroke starts and ends soft and is strongest
    in the middle, mimicking a human hand.

    The engine's brush 5 (压感v3) expects pressure values in the range
    0..8 (alpha = pressure / 8). We therefore scale the conceptual 0..1
    pressure up to 0..8 when emitting the point stream. brush 0 (马克笔)
    does not read pressure, so we still emit a 0 for every point to keep
    the array layout uniform (the engine ignores the trailing value).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import List, Dict, Any

# Canvas the engine renders onto (see engine.js: const e=640, t=480).
ENGINE_W = 640
ENGINE_H = 480

# Pressure scaling: engine brush 5 divides pressure by 8 to get alpha.
PRESSURE_SCALE = 8.0

# Background colour used by the original drawing app's first palette entry.
DEFAULT_BACKGROUND = "#f8ecdb"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_hex(r: float, g: float, b: float) -> str:
    """Convert 0..1 floats to #RRGGBB."""
    r = max(0, min(255, int(round(r * 255))))
    g = max(0, min(255, int(round(g * 255))))
    b = max(0, min(255, int(round(b * 255))))
    return f"#{r:02x}{g:02x}{b:02x}"


def pick_brush_id(radius: float) -> int:
    """Spec: thin lines use brush 5 (压感v3), broad strokes use brush 0 (马克笔)."""
    return 5 if radius < 3 else 0


def bell_pressure(t: float) -> float:
    """Bell-shaped pressure profile in [0, 1].

    t in [0, 1]. Peaks at t=0.5 with value 1.0, falls off smoothly to ~0.15
    at the ends. Uses a smoothstep-like curve so the ramp looks natural.
    """
    # raised-cosine bell: 0.15 + 0.85 * 0.5*(1 - cos(2*pi*t))
    return 0.15 + 0.85 * 0.5 * (1.0 - math.cos(2.0 * math.pi * t))


def interpolate_stroke(x0: float, y0: float, x1: float, y1: float,
                       n_points: int) -> List[tuple]:
    """Linear interpolation between two endpoints. Returns [(x, y, t), ...]."""
    n_points = max(2, n_points)
    pts = []
    for i in range(n_points):
        t = i / (n_points - 1)
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t
        pts.append((x, y, t))
    return pts


def stroke_point_count(x0: float, y0: float, x1: float, y1: float,
                       radius: float) -> int:
    """How many interpolated points a stroke should have.

    Longer strokes get more samples; very short strokes still get at least 5
    so the pressure curve has room to breathe.
    """
    dx = (x1 - x0) * ENGINE_W
    dy = (y1 - y0) * ENGINE_H
    length_px = math.hypot(dx, dy)
    # ~1 sample every 6 px, clamped to [5, 24]
    n = int(length_px / 6.0)
    return max(5, min(24, n))


# ---------------------------------------------------------------------------
# Stroke deduplication & merging (performance optimisation)
# ---------------------------------------------------------------------------

# Thresholds for considering two strokes "similar enough" to merge/drop.
# All coordinates are normalised to 0..1, so 0.01 = ~6px on a 640px canvas.
DEDUP_POS_THRESHOLD = 0.015   # ~10px positional difference
DEDUP_COLOR_THRESHOLD = 0.04  # ~10/255 per channel
DEDUP_RADIUS_THRESHOLD = 1.5  # px


def _stroke_signature(a: Dict[str, float]) -> tuple:
    """A hashable signature capturing a stroke's geometric + colour identity.

    Two strokes with the same signature are candidates for deduplication.
    We quantise the coordinates to a coarse grid so that strokes that are
    "almost the same" still collide.
    """
    # Quantise start/end to a 64x64 grid (0..1 -> 0..63)
    gx0 = int(a["x_start"] * 63)
    gy0 = int(a["y_start"] * 63)
    gx1 = int(a["x_end"] * 63)
    gy1 = int(a["y_end"] * 63)
    # Quantise colour to 16 levels per channel
    cr = int(a["color_r"] * 15)
    cg = int(a["color_g"] * 15)
    cb = int(a["color_b"] * 15)
    # Quantise radius to nearest 2px
    rr = int(a["brush_radius"] / 2)
    return (gx0, gy0, gx1, gy1, cr, cg, cb, rr)


def _strokes_similar(a: Dict[str, float], b: Dict[str, float]) -> bool:
    """Fine-grained similarity check (used after signature collision)."""
    dx0 = abs(a["x_start"] - b["x_start"])
    dy0 = abs(a["y_start"] - b["y_start"])
    dx1 = abs(a["x_end"] - b["x_end"])
    dy1 = abs(a["y_end"] - b["y_end"])
    if max(dx0, dy0, dx1, dy1) > DEDUP_POS_THRESHOLD:
        return False
    dr = abs(a["color_r"] - b["color_r"])
    dg = abs(a["color_g"] - b["color_g"])
    db = abs(a["color_b"] - b["color_b"])
    if max(dr, dg, db) > DEDUP_COLOR_THRESHOLD:
        return False
    if abs(a["brush_radius"] - b["brush_radius"]) > DEDUP_RADIUS_THRESHOLD:
        return False
    return True


def deduplicate_strokes(actions: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Remove near-duplicate strokes and merge collinear same-colour strokes.

    Two strategies:
      1. Exact-ish dedup: drop a stroke if a very similar stroke (same
         quantised position + colour + radius) already exists.
      2. Chain merging: if stroke B starts where stroke A ended and they
         share colour + radius, merge B into A (extend A's endpoint to B's
         endpoint) and drop B. This reduces stroke count for long curves
         that the model painted as many small segments.

    Returns a new list; the input is not mutated.
    """
    if len(actions) <= 1:
        return list(actions)

    seen_sigs: dict = {}
    deduped: List[Dict[str, float]] = []

    for a in actions:
        sig = _stroke_signature(a)
        # Strategy 1: exact-ish dedup
        is_dup = False
        if sig in seen_sigs:
            for idx in seen_sigs[sig]:
                if _strokes_similar(deduped[idx], a):
                    is_dup = True
                    break
        if is_dup:
            continue

        # Strategy 2: chain merge — does the previous stroke end where this
        # one starts, with matching colour + radius?
        if deduped:
            prev = deduped[-1]
            if (abs(prev["x_end"] - a["x_start"]) < DEDUP_POS_THRESHOLD and
                abs(prev["y_end"] - a["y_start"]) < DEDUP_POS_THRESHOLD and
                abs(prev["color_r"] - a["color_r"]) < DEDUP_COLOR_THRESHOLD and
                abs(prev["color_g"] - a["color_g"]) < DEDUP_COLOR_THRESHOLD and
                abs(prev["color_b"] - a["color_b"]) < DEDUP_COLOR_THRESHOLD and
                abs(prev["brush_radius"] - a["brush_radius"]) < DEDUP_RADIUS_THRESHOLD):
                # Merge: extend prev's endpoint to a's endpoint
                prev = dict(prev)  # copy to avoid mutating deduped entry
                prev["x_end"] = a["x_end"]
                prev["y_end"] = a["y_end"]
                deduped[-1] = prev
                continue

        # Keep this stroke
        seen_sigs.setdefault(sig, []).append(len(deduped))
        deduped.append(a)

    return deduped


# ---------------------------------------------------------------------------
# Core transform
# ---------------------------------------------------------------------------

def transform(actions: List[Dict[str, float]],
              background: str = DEFAULT_BACKGROUND,
              start_with_background: bool = True,
              dedup: bool = False) -> List[List[Any]]:
    """Convert raw model actions into the engine's flat instruction stream.

    Parameters
    ----------
    actions : list of action dicts (from model/inference.py)
    background : hex colour string for the leading background instruction
    start_with_background : if True, emit a leading ["background", ...] op
    dedup : if True, run stroke deduplication + chain merging before
            converting. This reduces the instruction count (faster
            rendering) without changing the output JSON format.
    """
    if dedup:
        original_count = len(actions)
        actions = deduplicate_strokes(actions)
        if len(actions) < original_count:
            print(f"[transform] dedup: {original_count} -> {len(actions)} strokes "
                  f"({original_count - len(actions)} removed)")

    instructions: List[List[Any]] = []

    if start_with_background:
        instructions.append(["background", background])

    for a in actions:
        x0 = a["x_start"]
        y0 = a["y_start"]
        x1 = a["x_end"]
        y1 = a["y_end"]
        r = a["color_r"]
        g = a["color_g"]
        b = a["color_b"]
        alpha = float(a.get("color_a", 1.0))
        radius = float(a["brush_radius"])

        # 1) colour + alpha
        instructions.append(["colour", to_hex(r, g, b)])
        instructions.append(["alpha", round(max(0.0, min(1.0, alpha)), 4)])

        # 2) brush width (engine uses the same value as the model radius)
        instructions.append(["width", round(radius, 3)])

        # 3) brush selection
        brush_id = pick_brush_id(radius)

        # 4) interpolate + pressure
        n_pts = stroke_point_count(x0, y0, x1, y1, radius)
        pts = interpolate_stroke(x0, y0, x1, y1, n_pts)

        # Build the flat point array. The engine's brush 0 (马克笔) reads
        # 2 values per point (x, y); brush 5 (压感v3) reads 3 (x, y, pressure).
        # We must match the stride or the brush misinterprets the data.
        flat: List[float] = [brush_id]
        for (px, py, t) in pts:
            if brush_id == 5:
                p = bell_pressure(t) * PRESSURE_SCALE  # 0..8
                flat.extend([round(px * ENGINE_W, 2),
                             round(py * ENGINE_H, 2),
                             round(p, 4)])
            else:
                # brush 0: no pressure, just x, y
                flat.extend([round(px * ENGINE_W, 2),
                             round(py * ENGINE_H, 2)])
        instructions.append(["line"] + flat)

    return instructions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Convert raw strokes -> engine JSON.")
    ap.add_argument("--input", default="raw_strokes.json",
                    help="Input raw strokes JSON (from model/inference.py).")
    ap.add_argument("--output", default="output_strokes.json",
                    help="Output engine JSON path.")
    ap.add_argument("--background", default=DEFAULT_BACKGROUND,
                    help="Background colour, e.g. #f8ecdb.")
    ap.add_argument("--no-background", action="store_true",
                    help="Do not emit a leading background instruction.")
    ap.add_argument("--dedup", action="store_true",
                    help="Enable stroke deduplication + chain merging "
                         "(reduces instruction count, faster rendering).")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        actions = json.load(f)
    print(f"[transform] loaded {len(actions)} raw actions from {args.input}")

    instructions = transform(
        actions,
        background=args.background,
        start_with_background=not args.no_background,
        dedup=args.dedup,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(instructions, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[transform] wrote {len(instructions)} instructions -> {args.output}")

    # quick stats
    n_line = sum(1 for i in instructions if i[0] == "line")
    n_colour = sum(1 for i in instructions if i[0] == "colour")
    print(f"[transform] stats: {n_line} line ops, {n_colour} colour ops")


if __name__ == "__main__":
    main()
