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
# Core transform
# ---------------------------------------------------------------------------

def transform(actions: List[Dict[str, float]],
              background: str = DEFAULT_BACKGROUND,
              start_with_background: bool = True) -> List[List[Any]]:
    """Convert raw model actions into the engine's flat instruction stream."""
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

        # Build the flat point array. For brush 0 (marker) the engine reads
        # 2 values per point; for brush 5 (压感v3) it reads 3 (x, y, p).
        # We always emit 3 values per point; brush 0 simply ignores the
        # trailing pressure value (the engine slices by the brush's stride).
        flat: List[float] = [brush_id]
        for (px, py, t) in pts:
            p = bell_pressure(t) * PRESSURE_SCALE  # 0..8
            flat.extend([round(px * ENGINE_W, 2),
                         round(py * ENGINE_H, 2),
                         round(p, 4)])
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
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        actions = json.load(f)
    print(f"[transform] loaded {len(actions)} raw actions from {args.input}")

    instructions = transform(
        actions,
        background=args.background,
        start_with_background=not args.no_background,
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
