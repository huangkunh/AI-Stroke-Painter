# 基础使用教程

本教程带你从零开始，完成第一次图片转笔画绘画的全流程。

## 前置准备

### 1. 安装 Python 依赖

```bash
# 最小安装（仅 Lite 模式，推荐新手）
pip install opencv-python numpy pillow

# 完整安装（含 RL 模式，需要 PyTorch）
pip install opencv-python numpy pillow torch torchvision
```

### 2. 克隆仓库

```bash
git clone https://github.com/huangkunh/AI-Stroke-Painter.git
cd AI-Stroke-Painter
```

## 三步完成绘画

### 步骤 1：模型推理

将图片转化为原始笔画数据：

```bash
python model/inference.py \
  --image assets/sample_cat.jpg \
  --mode lite \
  --max-steps 500 \
  --out raw_strokes.json
```

**参数说明**：
- `--image`：输入图片路径
- `--mode`：推理模式（`lite`/`rl`/`auto`）
- `--max-steps`：最大笔画数（100-2000）
- `--out`：输出文件路径

**输出**：`raw_strokes.json`，包含笔画动作数组。

### 步骤 2：数据转换

将原始笔画转换为渲染引擎可识别的 JSON：

```bash
python converter/transform.py \
  --input raw_strokes.json \
  --output output_strokes.json \
  --background "#f8ecdb"
```

**可选参数**：
- `--dedup`：启用笔画去重（减少指令数量）
- `--no-background`：不输出背景指令

### 步骤 3：浏览器预览

启动本地服务器并打开预览页面：

```bash
python -m http.server 8000
```

打开浏览器访问 `http://localhost:8000/renderer/index.html`：

1. 点击「📁 加载 JSON」
2. 选择 `output_strokes.json`
3. 点击「▶ 播放」观看逐笔绘画

## 播放器控制

| 按钮 | 功能 | 快捷键 |
|------|------|--------|
| ▶ 播放/⏸ 暂停 | 播放或暂停绘画 | Space |
| ⏭ 单步 | 绘制下一条指令 | → |
| ⏮ 重置 | 清空画布重新开始 | R |
| 🗑 清空画布 | 清空并卸载 JSON | C |
| ⊞ 显示网格 | 切换参考网格 | G |
| 💾 导出 PNG | 下载当前画布 | E |
| 速度下拉框 | 0.5x/1x/2x/4x | - |

## 常见问题

### Q: 播放时画面空白？

**A**: 检查：
1. JSON 文件是否为数组格式
2. 浏览器控制台是否有错误
3. 尝试 `--mode lite` 重新生成

### Q: 推理速度很慢？

**A**: 
- 降低 `--max-steps`（如 200）
- 降低 `--size`（如 256）
- 使用 `--mode lite`（无需 PyTorch）

### Q: 如何选择图片？

**A**: 
- **照片**：适合人物、动物、风景
- **插画**：适合扁平化设计、卡通
- **线稿**：适合素描、简笔画

系统会自动识别图片类型并调整笔画策略。

## 下一步

- [高级功能教程](./advanced.md)
- [故障排除指南](./troubleshooting.md)
- [API 使用指南](../api.md)
