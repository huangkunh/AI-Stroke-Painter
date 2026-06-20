#!/usr/bin/env python3
"""
Serverless API: combined pipeline (image -> raw strokes -> engine JSON).

This is a convenience endpoint that runs both inference and transformation
in a single request. Useful for clients that want the final engine-ready
JSON without making two round trips.

Supported runtimes: Vercel / Netlify / Lambda / CLI (same as infer.py).

Request (POST JSON):
  {
    "image": "<base64-encoded image bytes>",
    "mode": "lite",                  # optional
    "max_steps": 400,                # optional
    "size": 512,                     # optional
    "background": "#f8ecdb",         # optional
    "dedup": false                   # optional
  }

Response (200 JSON):
  {
    "instructions": [...],
    "stroke_count": N,
    "instruction_count": M,
    "mode": "lite",
    "elapsed_ms": 1234
  }
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
import traceback
from typing import Any, Dict

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_MODEL_DIR = os.path.join(_PROJECT_ROOT, "model")
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)

# Reuse rate limiter and auth from infer.py
from infer import _check_rate_limit, _check_api_key, _get_client_ip  # noqa: E402


def _run_pipeline(image_b64: str, mode: str, max_steps: int, size: int,
                  background: str, dedup: bool) -> Dict:
    """Run the full pipeline: image -> strokes -> engine instructions."""
    import numpy as np
    from PIL import Image
    import inference as inf
    from converter.transform import transform

    # Decode image
    image_bytes = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0

    # Inference
    if mode == "rl":
        actions = inf.run_rl_inference(
            arr,
            os.path.join(_MODEL_DIR, "pretrained", "actor_final.pth"),
            max_steps, "cpu"
        )
    else:
        actions = inf.run_lite_inference(arr, max_steps=max_steps)

    # Transform
    instructions = transform(
        actions,
        background=background,
        start_with_background=True,
        dedup=dedup
    )

    return {
        "instructions": instructions,
        "stroke_count": len(actions),
        "instruction_count": len(instructions),
        "mode": mode
    }


def _handle(body: Dict, headers: Dict, client_ip: str) -> tuple:
    """Process the request body and return (status_code, response_dict)."""
    if not _check_rate_limit(client_ip):
        return 429, {"error": "Rate limit exceeded. Try again later."}

    if not _check_api_key(headers):
        return 401, {"error": "Invalid or missing API key."}

    if not body or "image" not in body:
        return 400, {"error": "Missing 'image' field (base64-encoded image bytes)."}

    image_b64 = body["image"]
    mode = body.get("mode", os.environ.get("ASP_DEFAULT_MODE", "lite"))
    max_steps = int(body.get("max_steps", os.environ.get("ASP_DEFAULT_MAX_STEPS", "400")))
    size = int(body.get("size", os.environ.get("ASP_DEFAULT_SIZE", "512")))
    background = body.get("background", os.environ.get("ASP_DEFAULT_BACKGROUND", "#f8ecdb"))
    dedup = bool(body.get("dedup", False))

    if mode not in ("lite", "rl", "auto"):
        return 400, {"error": f"Invalid mode '{mode}'. Use lite/rl/auto."}
    if max_steps < 1 or max_steps > 2000:
        return 400, {"error": "max_steps must be between 1 and 2000."}
    if size < 64 or size > 1024:
        return 400, {"error": "size must be between 64 and 1024."}

    t0 = time.time()
    try:
        result = _run_pipeline(image_b64, mode, max_steps, size, background, dedup)
    except Exception as e:
        traceback.print_exc()
        return 500, {"error": f"Pipeline failed: {str(e)}"}

    result["elapsed_ms"] = int((time.time() - t0) * 1000)
    return 200, result


# ---------------------------------------------------------------------------
# Runtime adapters
# ---------------------------------------------------------------------------

def handler(request):
    """Vercel Python runtime entry point."""
    try:
        body = request.json if hasattr(request, 'json') else json.loads(request.body or "{}")
    except Exception:
        body = {}
    headers = dict(request.headers) if hasattr(request, 'headers') else {}
    client_ip = _get_client_ip(request)
    status, resp = _handle(body, headers, client_ip)
    from http import HTTPStatus
    return (json.dumps(resp, ensure_ascii=False), status,
            {"Content-Type": "application/json"})


def lambda_handler(event, context):
    """AWS Lambda (API Gateway proxy) entry point."""
    body_str = event.get("body", "{}")
    if event.get("isBase64Encoded"):
        body_str = base64.b64decode(body_str).decode("utf-8")
    try:
        body = json.loads(body_str) if body_str else {}
    except json.JSONDecodeError:
        return {"statusCode": 400, "body": json.dumps({"error": "Invalid JSON body."})}
    headers = event.get("headers", {})
    client_ip = _get_client_ip(event)
    status, resp = _handle(body, headers, client_ip)
    return {"statusCode": status, "body": json.dumps(resp, ensure_ascii=False),
            "headers": {"Content-Type": "application/json"}}


# Netlify uses the same signature as Lambda
netlify_handler = lambda_handler


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="AI Stroke Painter pipeline API (local mode)")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--mode", default="lite", choices=["lite", "rl", "auto"])
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--background", default="#f8ecdb")
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--out", default="-")
    args = ap.parse_args()

    with open(args.image, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("ascii")

    status, resp = _handle(
        {"image": image_b64, "mode": args.mode, "max_steps": args.max_steps,
         "size": args.size, "background": args.background, "dedup": args.dedup},
        {}, "127.0.0.1"
    )
    out = json.dumps(resp, ensure_ascii=False, indent=2)
    if args.out == "-":
        print(out)
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"Wrote {args.out} (status {status})")
