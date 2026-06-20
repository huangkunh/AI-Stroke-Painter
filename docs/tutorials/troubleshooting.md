# 故障排除指南

## 常见问题

### 1. `ModuleNotFoundError: No module named 'cv2'`

**原因**：未安装 OpenCV。

**解决**：
```bash
pip install opencv-python
# 或无头环境（服务器/Docker）
pip install opencv-python-headless
```

### 2. `ModuleNotFoundError: No module named 'torch'`

**原因**：未安装 PyTorch，RL 模式不可用。

**解决**：
```bash
# CPU 版本（推荐）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 或直接使用 lite 模式（无需 PyTorch）
python model/inference.py --image input.jpg --mode lite
```

### 3. 权重下载失败

**原因**：Google Drive 配额限制或网络问题。

**解决**：
1. 手动下载：访问 [ICCV2019-LearningToPaint](https://github.com/hzwer/ICCV2019-LearningToPaint)
2. 下载 `actor_final.pth`
3. 放到 `model/pretrained/actor_final.pth`
4. 验证：`bash model/download_weights.sh --check`
5. 或直接使用 `--mode lite`

### 4. 浏览器无法加载 JSON（CORS 错误）

**原因**：直接用 `file://` 协议打开 HTML。

**解决**：必须通过 HTTP 服务器访问：
```bash
python -m http.server 8000
# 然后访问 http://localhost:8000/renderer/index.html
```

### 5. Canvas 渲染空白

**排查步骤**：
1. 打开浏览器控制台（F12），查看是否有错误
2. 确认 JSON 文件是数组格式：`python -c "import json; d=json.load(open('output_strokes.json')); print(type(d), len(d))"`
3. 尝试重新生成：`python model/inference.py --image input.jpg --mode lite --max-steps 100`
4. 检查 JSON 内容：第一条应该是 `["background", ...]`

### 6. 推理速度很慢

**优化方案**：
- 降低 `--max-steps`（如 200）
- 降低 `--size`（如 256）
- 使用 `--mode lite`（比 RL 快 10-100 倍）
- 启用 `--dedup` 减少笔画数量

### 7. 绘画效果不理想

**调整建议**：
- **细节不够**：增加 `--max-steps`（如 800）
- **颜色不准**：检查原图是否被正确加载
- **笔画太粗**：系统根据图片类型自动选择，可尝试不同图片
- **空白区域多**：增加 `--max-steps` 或换更简单的图片

### 8. Docker 构建失败

**PyTorch 下载超时**：
```dockerfile
# 在 Dockerfile 中注释掉 PyTorch 安装行，仅用 Lite 模式
# RUN pip install torch torchvision ...
```

**OpenCV 缺少系统库**：
```dockerfile
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0
```

### 9. API 返回 401 Unauthorized

**原因**：设置了 `ASP_API_KEY` 但请求未携带密钥。

**解决**：
```bash
curl -H "X-API-Key: your-key" ...
```

### 10. API 返回 429 Too Many Requests

**原因**：超过速率限制（默认 60 次/分钟）。

**解决**：
- 等待一分钟后重试
- 或设置环境变量提高限制：`ASP_RATE_LIMIT=120`

## 获取帮助

如果以上方法都无法解决你的问题：

1. 查看 [GitHub Issues](https://github.com/huangkunh/AI-Stroke-Painter/issues)
2. 提交新 Issue，包含：
   - 错误信息（完整堆栈）
   - 操作系统
   - Python 版本
   - 复现步骤
3. 查看 [基础教程](./basic.md) 和 [高级教程](./advanced.md)
