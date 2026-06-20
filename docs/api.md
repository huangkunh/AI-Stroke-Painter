# AI Stroke Painter REST API

The AI Stroke Painter exposes two serverless API endpoints for programmatic
access to the image-to-stroke pipeline.

## Base URL

- **Vercel**: `https://your-app.vercel.app/api`
- **Netlify**: `https://your-app.netlify.app/.netlify/functions`
- **Local**: `http://localhost:3000/api` (via `vercel dev` or `netlify dev`)

## Authentication

If the `ASP_API_KEY` environment variable is set, all requests must include:

```
X-API-Key: your-secret-key
```

Requests without a valid key receive `401 Unauthorized`.

## Rate Limiting

Default: 60 requests/minute per IP (configurable via `ASP_RATE_LIMIT`).
Exceeding the limit returns `429 Too Many Requests`.

---

## POST /api/infer

Convert an image to raw stroke data.

### Request

```json
{
  "image": "<base64-encoded image bytes>",
  "mode": "lite",
  "max_steps": 400,
  "size": 512
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | string | *required* | Base64-encoded image (JPG/PNG/BMP/WEBP) |
| `mode` | string | `"lite"` | Inference backend: `lite`, `rl`, or `auto` |
| `max_steps` | int | `400` | Maximum number of strokes (1-2000) |
| `size` | int | `512` | Internal canvas size (64-1024) |

### Response (200)

```json
{
  "strokes": [
    {"x_start": 0.5, "y_start": 0.3, "x_end": 0.6, "y_end": 0.4,
     "color_r": 0.8, "color_g": 0.2, "color_b": 0.1, "color_a": 0.7,
     "brush_radius": 12.0}
  ],
  "count": 400,
  "mode": "lite",
  "elapsed_ms": 1234
}
```

### Errors

| Status | Cause |
|--------|-------|
| 400 | Missing/invalid `image` field |
| 401 | Missing/invalid API key |
| 413 | Image too large (>10MB) |
| 429 | Rate limit exceeded |
| 500 | Inference error (see `error` field) |

### Example (curl)

```bash
# Encode image to base64
IMAGE_B64=$(base64 -w0 sample.jpg)

curl -X POST https://your-app.vercel.app/api/infer \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d "{\"image\": \"$IMAGE_B64\", \"mode\": \"lite\", \"max_steps\": 300}" \
  -o strokes.json
```

### Example (Python)

```python
import base64, requests

with open("sample.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

resp = requests.post("https://your-app.vercel.app/api/infer",
    json={"image": img_b64, "mode": "lite", "max_steps": 300},
    headers={"X-API-Key": "your-key"})
strokes = resp.json()["strokes"]
```

### Example (JavaScript)

```javascript
const fileInput = document.querySelector('input[type=file]');
const file = fileInput.files[0];
const reader = new FileReader();
reader.onload = async () => {
  const b64 = reader.result.split(',')[1];
  const resp = await fetch('/api/infer', {
    method: 'POST',
    headers: {'Content-Type': 'application/json', 'X-API-Key': 'your-key'},
    body: JSON.stringify({image: b64, mode: 'lite', max_steps: 300})
  });
  const { strokes } = await resp.json();
  console.log('Got', strokes.length, 'strokes');
};
reader.readAsDataURL(file);
```

---

## POST /api/transform

Convert raw strokes to engine-renderable JSON instructions.

### Request

```json
{
  "strokes": [...],
  "background": "#f8ecdb",
  "dedup": false
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `strokes` | array | *required* | Raw stroke dicts from `/api/infer` |
| `background` | string | `"#f8ecdb"` | Background colour (hex) |
| `dedup` | bool | `false` | Enable stroke deduplication |

### Response (200)

```json
{
  "instructions": [
    ["background", "#f8ecdb"],
    ["colour", "#d20000"],
    ["width", 10],
    ["alpha", 0.8],
    ["line", 0, 100, 100, 200, 200, 300, 100]
  ],
  "count": 1601,
  "elapsed_ms": 5
}
```

### Example (full pipeline)

```bash
# Step 1: infer
curl -X POST .../api/infer -d '{"image":"..."}' -o strokes.json

# Step 2: transform
curl -X POST .../api/transform \
  -H "Content-Type: application/json" \
  -d "{\"strokes\": $(cat strokes.json | jq .strokes)}" \
  -o instructions.json
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ASP_DEFAULT_MAX_STEPS` | `400` | Default max strokes if not specified |
| `ASP_DEFAULT_MODE` | `lite` | Default inference mode |
| `ASP_DEFAULT_SIZE` | `512` | Default canvas size |
| `ASP_DEFAULT_BACKGROUND` | `#f8ecdb` | Default background colour |
| `ASP_API_KEY` | *(unset)* | If set, enables API key authentication |
| `ASP_RATE_LIMIT` | `60` | Requests per minute per IP |

---

## Local Development

```bash
# Vercel
npm i -g vercel
vercel dev

# Netlify
npm i -g netlify-cli
netlify dev

# Or test API functions directly
python api/infer.py --image assets/sample_cat.jpg --mode lite --max-steps 100
python api/transform.py --input /tmp/strokes.json --dedup
```
