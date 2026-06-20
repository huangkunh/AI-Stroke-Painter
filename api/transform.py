#!/usr/bin/env python3
"""
Serverless API: raw strokes -> engine JSON (data transformation).

Same runtime adapters as infer.py (Vercel / Netlify / Lambda / CLI).

Environment variables:
  ASP_DEFAULT_BACKGROUND  (default #f8ecdb)  background colour
  ASP_API_KEY             (optional)         if set, requests must send X-API-Key
  ASP_RATE_LIMIT          (default 60)       max requests per minute per IP

Request (POST JSON):
  {
    "strokes": [...],                # raw action dicts from infer.py
    "background": "#f8ecdb",         # optional
    "dedup": false                   # optional, enable stroke deduplication
  }

Response (200 JSON):
  { "instructions": [...], "count": N, "elapsed_ms": 123 }
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from typing import Any, Dict

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Rate limiter (shared pattern with infer.py)
_rate_store: Dict[str, list] = {}
_RATE_LIMIT = int(os.environ.get("ASP_RATE_LIMIT", "60"))


def _check_rate_limit(client_ip: str) -> bool:
    now = time.time()
    window = 60.0
    if client_ip not in _rate_store:
        _rate_store[client_ip] = [now]
        return True
    _rate_store[client_ip] = [t for t in _rate_store[client_ip] if now - t < window]
    if len(_rate_store[client_ip]) >= _RATE_LIMIT:
        return False
    _rate_store[client_ip].append(now)
    return True


def _check_api_key(headers: Dict[str, str]) -> bool:
    expected = os.environ.get("ASP_API_KEY", "")
    if not expected:
        return True
    provided = headers.get("x-api-key") or headers.get("X-API-Key", "")
    return provided == expected


def _get_client_ip(event_or_request: Any) -> str:
    headers = {}
    if isinstance(event_or_request, dict):
        headers = event_or_request.get("headers") or {}
    else:
        headers = getattr(event_or_request, "headers", {}) or {}
    for key in ("x-forwarded-for", "x-real-ip", "client-ip", "remote-addr"):
        val = headers.get(key) or headers.get(key.title()) or headers.get(key.upper())
        if val:
            return val.split(",")[0].strip()
    return "unknown"


def _handle(body: Dict, headers: Dict, client_ip: str) -> tuple:
    if not _check_rate_limit(client_ip):
        return 429, {"error": "Rate limit exceeded. Try again later."}
    if not _check_api_key(headers):
        return 401, {"error": "Invalid or missing API key."}

    if not body or "strokes" not in body:
        return 400, {"error": "Missing 'strokes' field (array of action dicts)."}

    strokes = body["strokes"]
    if not isinstance(strokes, list):
        return 400, {"error": "'strokes' must be an array."}

    background = body.get("background", os.environ.get("ASP_DEFAULT_BACKGROUND", "#f8ecdb"))
    dedup = bool(body.get("dedup", False))

    try:
        from converter.transform import transform
        t0 = time.time()
        instructions = transform(strokes, background=background,
                                 start_with_background=True, dedup=dedup)
        return 200, {
            "instructions": instructions,
            "count": len(instructions),
            "elapsed_ms": int((time.time() - t0) * 1000)
        }
    except Exception as e:
        traceback.print_exc()
        return 500, {"error": f"Transform failed: {str(e)}"}


# Runtime adapters (same pattern as infer.py)
def handler(request, context=None):
    if isinstance(request, dict):
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
    body_str = event.get("body", "{}")
    if event.get("isBase64Encoded"):
        import base64
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


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="AI Stroke Painter transform API (local mode)")
    ap.add_argument("--input", required=True, help="Input raw strokes JSON (array or {strokes:[...]})")
    ap.add_argument("--background", default="#f8ecdb")
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--out", default="-")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    # Accept both bare arrays and {"strokes": [...]} wrappers
    if isinstance(loaded, list):
        strokes = loaded
    elif isinstance(loaded, dict) and "strokes" in loaded:
        strokes = loaded["strokes"]
    else:
        print("Error: input must be a JSON array or {\"strokes\": [...]}")
        sys.exit(1)
    status, resp = _handle({"strokes": strokes, "background": args.background, "dedup": args.dedup},
                           {}, "127.0.0.1")
    out = json.dumps(resp, ensure_ascii=False, indent=2)
    if args.out == "-":
        print(out)
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"Wrote {args.out} (status {status})")
