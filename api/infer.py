#!/usr/bin/env python3
"""
Serverless API: image -> raw strokes (model inference).

Supported runtimes:
  - Vercel Python (@vercel/python): entry point is `handler(request)`
  - Netlify Functions: entry point is `handler(event, context)`
  - AWS Lambda (via API Gateway proxy): entry point is `lambda_handler(event, context)`

This file auto-detects the runtime and dispatches accordingly.

Environment variables:
  ASP_DEFAULT_MAX_STEPS  (default 400)  max strokes to emit
  ASP_DEFAULT_MODE        (default lite) inference backend
  ASP_DEFAULT_SIZE        (default 512)  internal canvas size
  ASP_API_KEY             (optional)     if set, requests must send X-API-Key
  ASP_RATE_LIMIT          (default 60)   max requests per minute per IP

Request (POST JSON):
  {
    "image": "<base64-encoded image bytes>",
    "mode": "lite" | "rl" | "auto",      # optional
    "max_steps": 400,                      # optional
    "size": 512                            # optional
  }

Response (200 JSON):
  { "strokes": [...], "count": N, "mode": "lite", "elapsed_ms": 123 }

Response (400/401/429 JSON):
  { "error": "message" }
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

# Make project modules importable when deployed as a serverless function.
# In Vercel/Netlify the CWD is the project root, so we add it to sys.path.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_MODEL_DIR = os.path.join(_PROJECT_ROOT, "model")
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)


# ---------------------------------------------------------------------------
# Simple in-memory rate limiter (per-IP, per-minute)
# ---------------------------------------------------------------------------
_rate_store: Dict[str, list] = {}
_RATE_LIMIT = int(os.environ.get("ASP_RATE_LIMIT", "60"))


def _check_rate_limit(client_ip: str) -> bool:
    """Return True if request is allowed, False if rate-limited."""
    now = time.time()
    window = 60.0
    if client_ip not in _rate_store:
        _rate_store[client_ip] = [now]
        return True
    # Prune old entries
    _rate_store[client_ip] = [t for t in _rate_store[client_ip] if now - t < window]
    if len(_rate_store[client_ip]) >= _RATE_LIMIT:
        return False
    _rate_store[client_ip].append(now)
    return True


def _check_api_key(headers: Dict[str, str]) -> bool:
    """Return True if API key is valid (or not required)."""
    expected = os.environ.get("ASP_API_KEY", "")
    if not expected:
        return True  # no key configured -> open access
    provided = headers.get("x-api-key") or headers.get("X-API-Key", "")
    return provided == expected


def _get_client_ip(event_or_request: Any) -> str:
    """Extract client IP from request/event headers."""
    headers = {}
    if isinstance(event_or_request, dict):
        headers = event_or_request.get("headers") or {}
    else:
        headers = getattr(event_or_request, "headers", {}) or {}
    # Headers may be lowercased
    for key in ("x-forwarded-for", "x-real-ip", "client-ip", "remote-addr"):
        val = headers.get(key) or headers.get(key.title()) or headers.get(key.upper())
        if val:
            return val.split(",")[0].strip()
    return "unknown"


# ---------------------------------------------------------------------------
# Warm-up: pre-import heavy dependencies at module load time (cold start)
# ---------------------------------------------------------------------------
# Serverless functions suffer from cold-start latency when the Python runtime
# must import numpy, PIL, cv2, and the inference module for the first time.
# By importing them at module level (which runs during cold start, before the
# first request is handled), we move that cost out of the request path.
_warmup_done = False
_warmup_error = None

def _warmup():
    """Pre-import heavy modules to reduce first-request latency."""
    global _warmup_done, _warmup_error
    if _warmup_done:
        return
    try:
        import numpy as np  # noqa: F401
        from PIL import Image  # noqa: F401
        import cv2  # noqa: F401
        import inference as inf  # noqa: F401
        # Touch the lite inference path with a tiny image to JIT-compile
        # the cv2.kmeans / Canny code paths.
        _tiny = np.zeros((32, 32, 3), dtype=np.float32)
        _tiny[:16, :16] = [0.8, 0.2, 0.1]
        _tiny[16:, 16:] = [0.1, 0.8, 0.2]
        inf.run_lite_inference(_tiny, max_steps=5)
        _warmup_done = True
    except Exception as e:
        _warmup_error = str(e)
        # Don't raise; the actual request will fail with a clearer error.

# Run warmup at module import (cold start)
_warmup()


# ---------------------------------------------------------------------------
# Core inference logic
# ---------------------------------------------------------------------------

def _run_inference(image_b64: str, mode: str, max_steps: int, size: int) -> Dict:
    """Decode image, run inference, return strokes dict."""
    import numpy as np
    from PIL import Image
    import inference as inf

    # Decode base64 image
    image_bytes = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0

    # Run inference (always falls back to lite if RL unavailable)
    if mode == "rl":
        actions = inf.run_rl_inference(arr, os.path.join(_MODEL_DIR, "pretrained", "actor_final.pth"),
                                        max_steps, "cpu")
    else:
        actions = inf.run_lite_inference(arr, max_steps=max_steps)

    return {"strokes": actions, "count": len(actions), "mode": mode}


# ---------------------------------------------------------------------------
# Request handler (shared logic)
# ---------------------------------------------------------------------------

def _handle(body: Dict, headers: Dict, client_ip: str) -> tuple:
    """Process the request body and return (status_code, response_dict)."""
    # Rate limit
    if not _check_rate_limit(client_ip):
        return 429, {"error": "Rate limit exceeded. Try again later."}

    # API key
    if not _check_api_key(headers):
        return 401, {"error": "Invalid or missing API key."}

    # Validate body
    if not body or "image" not in body:
        return 400, {"error": "Missing 'image' field (base64-encoded image bytes)."}

    image_b64 = body["image"]
    mode = body.get("mode", os.environ.get("ASP_DEFAULT_MODE", "lite"))
    max_steps = int(body.get("max_steps", os.environ.get("ASP_DEFAULT_MAX_STEPS", "400")))
    size = int(body.get("size", os.environ.get("ASP_DEFAULT_SIZE", "512")))

    if mode not in ("lite", "rl", "auto"):
        return 400, {"error": f"Invalid mode '{mode}'. Use lite/rl/auto."}
    if max_steps < 1 or max_steps > 2000:
        return 400, {"error": "max_steps must be between 1 and 2000."}
    if size < 64 or size > 1024:
        return 400, {"error": "size must be between 64 and 1024."}

    # Run inference
    t0 = time.time()
    try:
        result = _run_inference(image_b64, mode, max_steps, size)
        result["elapsed_ms"] = int((time.time() - t0) * 1000)
        return 200, result
    except Exception as e:
        traceback.print_exc()
        return 500, {"error": f"Inference failed: {str(e)}", "traceback": traceback.format_exc()[-500:]}


# ---------------------------------------------------------------------------
# Runtime adapters
# ---------------------------------------------------------------------------

def handler(request, context=None):
    """Vercel Python / Netlify Function entry point."""
    # Vercel: request is a Starlette Request with .body() async
    # Netlify: request is a dict with 'body', 'headers', 'httpMethod'
    if isinstance(request, dict):
        # Netlify style
        body_str = request.get("body", "{}")
        try:
            body = json.loads(body_str) if body_str else {}
        except json.JSONDecodeError:
            return {"statusCode": 400, "body": json.dumps({"error": "Invalid JSON body."})}
        headers = request.get("headers", {})
        client_ip = _get_client_ip(request)
        status, resp = _handle(body, headers, client_ip)
        return {"statusCode": status, "body": json.dumps(resp, ensure_ascii=False),
                "headers": {"Content-Type": "application/json"}}
    else:
        # Vercel style (sync wrapper) — body is bytes
        try:
            body = json.loads(request.body.decode("utf-8") if isinstance(request.body, bytes) else request.body)
        except Exception:
            body = {}
        headers = dict(request.headers)
        client_ip = _get_client_ip(request)
        status, resp = _handle(body, headers, client_ip)
        from starlette.responses import JSONResponse
        return JSONResponse(resp, status_code=status)


def lambda_handler(event, context):
    """AWS Lambda entry point (API Gateway proxy integration)."""
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


# ---------------------------------------------------------------------------
# CLI mode (for local testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="AI Stroke Painter inference API (local mode)")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--mode", default="lite", choices=["lite", "rl", "auto"])
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", default="-", help="Output file (- for stdout)")
    args = ap.parse_args()

    with open(args.image, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("ascii")

    status, resp = _handle(
        {"image": image_b64, "mode": args.mode, "max_steps": args.max_steps, "size": args.size},
        {}, "127.0.0.1"
    )
    out = json.dumps(resp, ensure_ascii=False, indent=2)
    if args.out == "-":
        print(out)
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"Wrote {args.out} (status {status})")
