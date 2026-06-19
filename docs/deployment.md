# 部署指南

本文档详细说明 AI Stroke Painter 在不同环境下的部署步骤。

---

## 目录

- [本地部署](#本地部署)
- [Docker 部署](#docker-部署)
- [服务器部署](#服务器部署)
- [云平台部署](#云平台部署)
- [生产环境配置](#生产环境配置)
- [故障排查](#故障排查)

---

## 本地部署

### 前置条件

- Python 3.8+
- （可选）Node.js 14+，用于本地预览服务器
- （可选）Git，用于克隆仓库

### 步骤

#### 1. 克隆仓库

```bash
git clone https://github.com/huangkunh/AI-Stroke-Painter.git
cd AI-Stroke-Painter
```

#### 2. 安装 Python 依赖

**最小安装**（仅 Lite 模式，推荐快速演示）：

```bash
pip install opencv-python numpy pillow
```

**完整安装**（含 RL 模式）：

```bash
pip install opencv-python numpy pillow torch torchvision
```

#### 3. （可选）下载预训练权重

```bash
bash model/download_weights.sh
```

如果下载失败（Google Drive 配额限制），可以：
- 手动从 [ICCV2019-LearningToPaint](https://github.com/hzwer/ICCV2019-LearningToPaint) 下载 `actor_final.pth`
- 放到 `model/pretrained/actor_final.pth`
- 或直接使用 `--mode lite` 无需权重

#### 4. 运行完整管线

```bash
# 步骤 1：模型推理
python model/inference.py --image assets/sample_cat.jpg --mode lite --max-steps 500

# 步骤 2：数据转换
python converter/transform.py --input raw_strokes.json --output output_strokes.json

# 步骤 3：启动预览服务器
python -m http.server 8000
```

打开浏览器访问 `http://localhost:8000/renderer/index.html`，点击「加载 JSON」选择 `output_strokes.json`。

---

## Docker 部署

### 前置条件

- Docker 20+
- Docker Compose（可选）

### 快速开始

#### 构建镜像

```bash
docker build -t ai-stroke-painter .
```

#### 运行预览服务器

```bash
docker run -d \
  --name asp-preview \
  -p 8000:8000 \
  ai-stroke-painter
```

打开浏览器访问 `http://localhost:8000/renderer/index.html`。

#### 在容器内运行推理

```bash
# 将本地图片挂载到容器，运行推理
docker run -it --rm \
  -v $(pwd)/assets:/workspace/assets \
  -v $(pwd)/output:/workspace/output \
  ai-stroke-painter \
  python model/inference.py --image assets/sample_cat.jpg --mode lite --out output/raw_strokes.json

# 运行转换
docker run -it --rm \
  -v $(pwd)/output:/workspace/output \
  ai-stroke-painter \
  python converter/transform.py --input output/raw_strokes.json --output output/output_strokes.json
```

#### Docker Compose

创建 `docker-compose.yml`：

```yaml
version: '3.8'
services:
  asp:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./assets:/workspace/assets
      - ./output:/workspace/output
    restart: unless-stopped
```

运行：

```bash
docker-compose up -d
```

---

## 服务器部署

### Nginx + Python 后端

适用于需要将 AI Stroke Painter 作为 Web 服务部署的场景。

#### 1. 服务器准备

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3 python3-pip nginx

# CentOS/RHEL
sudo yum install -y python3 python3-pip nginx
```

#### 2. 部署代码

```bash
sudo mkdir -p /opt/ai-stroke-painter
sudo chown $USER:$USER /opt/ai-stroke-painter
git clone https://github.com/huangkunh/AI-Stroke-Painter.git /opt/ai-stroke-painter
cd /opt/ai-stroke-painter
pip3 install --user opencv-python numpy pillow
```

#### 3. 配置 Nginx

创建 `/etc/nginx/conf.d/ai-stroke-painter.conf`：

```nginx
server {
    listen 80;
    server_name your-domain.com;

    root /opt/ai-stroke-painter;
    index renderer/index.html;

    location / {
        try_files $uri $uri/ =404;
    }

    # 静态资源缓存
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|json)$ {
        expires 1h;
        add_header Cache-Control "public, no-transform";
    }
}
```

#### 4. 启动

```bash
sudo nginx -t
sudo systemctl restart nginx
```

访问 `http://your-domain.com/renderer/index.html`。

---

## 云平台部署

### Vercel / Netlify（静态托管）

AI Stroke Painter 的前端是纯静态文件，可以直接部署到 Vercel 或 Netlify。

#### Vercel

```bash
npm i -g vercel
cd AI-Stroke-Painter
vercel --prod
```

配置 `vercel.json`：

```json
{
  "builds": [
    { "src": "renderer/**", "use": "@vercel/static" }
  ],
  "routes": [
    { "src": "/", "dest": "/renderer/index.html" }
  ]
}
```

> **注意**：云静态托管只能提供前端预览，推理和转换需要在本地运行后上传生成的 JSON。

### Google Cloud Run / AWS Lambda（带推理）

如果需要云端推理能力，可以使用 Docker 部署到 Cloud Run：

```bash
# Google Cloud Run
gcloud builds submit --tag gcr.io/your-project/ai-stroke-painter
gcloud run deploy ai-stroke-painter \
  --image gcr.io/your-project/ai-stroke-painter \
  --port 8000 \
  --allow-unauthenticated
```

### Hugging Face Spaces

创建 `app.py` 作为入口，使用 Gradio 包装推理流程：

```python
import gradio as gr
import subprocess

def paint(image):
    image.save("input.jpg")
    subprocess.run(["python", "model/inference.py", "--image", "input.jpg", "--mode", "lite"])
    subprocess.run(["python", "converter/transform.py", "--input", "raw_strokes.json"])
    return "output_strokes.json"

gr.Interface(fn=paint, inputs=gr.Image(), outputs=gr.File()).launch()
```

---

## 生产环境配置

### 性能优化

1. **使用 `--dedup` 减少指令数量**：

   ```bash
   python converter/transform.py --input raw_strokes.json --output output_strokes.json --dedup
   ```

2. **调整 `--max-steps`**：根据图片复杂度调整，简单图片 200-300 步，复杂图片 500-800 步。

3. **使用离屏预渲染**：在前端调用 `renderToOffscreen()` 预渲染完成笔画，减少主 Canvas 绘制压力。

### 安全配置

如果部署为公开服务，建议：

1. **限制上传文件大小**（Nginx 示例）：

   ```nginx
   client_max_body_size 10M;
   ```

2. **启用 HTTPS**：使用 Let's Encrypt 免费证书。

3. **速率限制**：防止滥用推理 API。

---

## 故障排查

### 问题：`ModuleNotFoundError: No module named 'cv2'`

```bash
pip install opencv-python
# 或无头环境
pip install opencv-python-headless
```

### 问题：权重下载失败

Google Drive 配额限制，手动下载：
1. 访问 https://github.com/hzwer/ICCV2019-LearningToPaint
2. 下载 `actor_final.pth`
3. 放到 `model/pretrained/actor_final.pth`
4. 或使用 `--mode lite`

### 问题：浏览器无法加载 JSON（CORS 错误）

必须通过 HTTP 服务器访问，不能直接 `file://` 打开：

```bash
python -m http.server 8000
# 然后访问 http://localhost:8000/renderer/index.html
```

### 问题：Canvas 渲染空白

检查：
1. JSON 文件是否为数组格式
2. 浏览器控制台是否有错误
3. 尝试 `--mode lite` 重新生成

### 问题：Docker 构建失败（PyTorch 下载超时）

Dockerfile 中 PyTorch 安装是可选的，构建失败不影响 Lite 模式。如需 RL 模式，可以：
1. 使用国内镜像源
2. 或在 Dockerfile 中注释掉 PyTorch 安装行，仅用 Lite 模式
