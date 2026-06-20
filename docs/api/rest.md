# REST API 使用指南

本指南介绍如何通过 REST API 使用 AI Stroke Painter 的云函数。

## 目录

- [认证](#认证)
- [速率限制](#速率限制)
- [POST /api/infer](#post-apiinfer)
- [POST /api/transform](#post-apitransform)
- [POST /api/pipeline](#post-apipipeline)
- [环境变量](#环境变量)
- [完整示例](#完整示例)

---

## 认证

如果设置了 `ASP_API_KEY` 环境变量，所有请求必须携带:

```
X-API-Key: your-secret-key
```

未携带有效密钥的请求返回 `401 Unauthorized`。

## 速率限制

默认: 60 请求/分钟/IP（可通过 `ASP_RATE_LIMIT` 配置）。
超限返回 `429 Too Many Requests`。

## POST /api/infer

将图片转换为原始笔画数据。

### 请求

```json
{
  "image": "<base64-encoded image bytes>",
  "mode": "lite",
  "max_steps": 400,
  "size": 512
}
```

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `image` | string | *必填* | Base64 编码的图片 |
| `mode` | string | `"lite"` | 推理模式: `lite`/`rl`/`auto` |
| `max_steps` | int | `400` | 最大笔画数 (1-2000) |
| `size` | int | `512` | 画布尺寸 (64-1024) |

### 响应 (200)

```json
{
  "strokes": [...],
  "count": 400,
  "mode": "lite",
  "elapsed_ms": 1234
}
```

### curl 示例

```bash
# 编码图片
IMAGE_B64=$(base64 -w 0 input.jpg)

curl -X POST https://your-app.vercel.app/api/infer \
  -H "Content-Type: application/json" \
  -d "{\"image\":\"$IMAGE_B64\",\"mode\":\"lite\",\"max_steps\":400}"
```

### Python 示例

```python
import requests
import base64

with open('input.jpg', 'rb') as f:
    image_b64 = base64.b64encode(f.read()).decode()

resp = requests.post('https://your-app.vercel.app/api/infer', json={
    'image': image_b64,
    'mode': 'lite',
    'max_steps': 400,
    'size': 512
})
data = resp.json()
print(f"Got {data['count']} strokes in {data['elapsed_ms']}ms")
```

### JavaScript 示例

```javascript
const fileInput = document.querySelector('input[type=file]');
const file = fileInput.files[0];
const reader = new FileReader();
reader.onload = async () => {
  const image_b64 = reader.result.split(',')[1];
  const resp = await fetch('https://your-app.vercel.app/api/infer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: image_b64, mode: 'lite', max_steps: 400 })
  });
  const data = await resp.json();
  console.log(`Got ${data.count} strokes`);
};
reader.readAsDataURL(file);
```

## POST /api/transform

将原始笔画数据转换为引擎 JSON 指令。

### 请求

```json
{
  "strokes": [...],
  "background": "#f8ecdb",
  "dedup": false
}
```

### 响应 (200)

```json
{
  "instructions": [...],
  "count": 1601,
  "elapsed_ms": 5
}
```

### 示例

```bash
curl -X POST https://your-app.vercel.app/api/transform \
  -H "Content-Type: application/json" \
  -d '{"strokes": [...], "background": "#f8ecdb", "dedup": true}'
```

## POST /api/pipeline

组合端点：一次调用完成推理 + 转换。

### 请求

```json
{
  "image": "<base64>",
  "mode": "lite",
  "max_steps": 400,
  "size": 512,
  "background": "#f8ecdb",
  "dedup": false
}
```

### 响应 (200)

```json
{
  "instructions": [...],
  "stroke_count": 400,
  "instruction_count": 1601,
  "mode": "lite",
  "elapsed_ms": 1234
}
```

### 示例

```python
import requests, base64

with open('input.jpg', 'rb') as f:
    image_b64 = base64.b64encode(f.read()).decode()

resp = requests.post('https://your-app.vercel.app/api/pipeline', json={
    'image': image_b64,
    'mode': 'lite',
    'max_steps': 400,
    'dedup': True
})
data = resp.json()
print(f"{data['stroke_count']} strokes -> {data['instruction_count']} instructions")
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ASP_DEFAULT_MAX_STEPS` | `400` | 默认最大笔画数 |
| `ASP_DEFAULT_MODE` | `lite` | 默认推理模式 |
| `ASP_DEFAULT_SIZE` | `512` | 默认画布尺寸 |
| `ASP_DEFAULT_BACKGROUND` | `#f8ecdb` | 默认背景色 |
| `ASP_API_KEY` | *(未设置)* | API 密钥（设置后启用认证） |
| `ASP_RATE_LIMIT` | `60` | 每分钟请求限制 |

## 完整示例

### Python 完整流程

```python
import requests
import base64
import json

# 1. 读取并编码图片
with open('input.jpg', 'rb') as f:
    image_b64 = base64.b64encode(f.read()).decode()

# 2. 调用 pipeline API
resp = requests.post('https://your-app.vercel.app/api/pipeline', json={
    'image': image_b64,
    'mode': 'lite',
    'max_steps': 400,
    'background': '#f8ecdb',
    'dedup': True
})

if resp.status_code == 200:
    data = resp.json()
    instructions = data['instructions']
    print(f"Generated {data['instruction_count']} instructions in {data['elapsed_ms']}ms")

    # 3. 保存到文件
    with open('output_strokes.json', 'w') as f:
        json.dump(instructions, f)
else:
    print(f"Error {resp.status_code}: {resp.json()['error']}")
```

### 错误处理

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 401 | API 密钥无效或缺失 |
| 429 | 速率限制超限 |
| 500 | 服务器内部错误 |

```python
resp = requests.post(url, json=payload)
if resp.status_code == 429:
    print("Rate limited, retry after 60s")
elif resp.status_code == 401:
    print("Invalid API key")
elif resp.status_code != 200:
    print(f"Error: {resp.json().get('error', 'unknown')}")
```
