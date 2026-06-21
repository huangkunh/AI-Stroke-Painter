# AI Stroke Painter

> 将输入图片转化为具有人类绘画时序的笔画数据，并通过自定义 Canvas 2D 引擎逐笔回放渲染。

本项目基于强化学习模型 [Learning-to-Paint](https://github.com/hzwer/ICCV2019-LearningToPaint)（ICCV 2019），实现了一条完整的 **图片 → AI 笔画推理 → 数据转换 → 浏览器逐笔回放** 管线。

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-optional-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/huangkunh/AI-Stroke-Painter/actions/workflows/ci.yml/badge.svg)](https://github.com/huangkunh/AI-Stroke-Painter/actions)

> 🌐 **在线 Demo**: [https://ai-stroke-painter.vercel.app](https://ai-stroke-painter.vercel.app) （部署后可访问）

---

## 目录

- [项目简介](#项目简介)
- [架构图](#架构图)
- [算法原理](#算法原理)
- [快速开始](#快速开始)
- [API 文档](#api-文档)
- [测试](#测试)
- [部署](#部署)
- [教程](#教程)
- [FAQ](#faq)
- [贡献](#贡献)
- [致谢](#致谢)
- [License](#license)

---

## 项目简介

AI Stroke Painter 分为三层：

| 层 | 目录 | 职责 |
|---|---|---|
| **模型推理端** | `model/` | 加载预训练 Agent，对目标图片运行强化学习推理循环，输出每一笔的动作参数 |
| **数据转换层** | `converter/` | 将模型输出的原始动作张量转换为渲染引擎可识别的扁平 JSON 指令流 |
| **渲染引擎端** | `renderer/` | 封装自定义 Canvas 2D 笔刷引擎，接收 JSON 指令并在画布上逐笔绘制 |

### 核心特性

- **双推理后端**：RL 模式（Learning-to-Paint 预训练权重）+ Lite 模式（纯 NumPy/OpenCV 启发式画家）
- **自动降级**：RL 权重缺失或加载失败时自动降级到 Lite 模式，管线永不中断
- **图片类型识别**：自动区分照片/线稿/插画，针对不同类型调整笔画策略
- **笔画去重**：转换层支持笔画去重和链式合并，减少指令数量加速渲染
- **离屏预渲染**：渲染引擎支持离屏 Canvas 预渲染，提升大笔画集回放性能
- **完整播放控制**：播放/暂停/单步/重置/速度调节/网格/导出 PNG
- **权重完整性校验**：下载脚本支持 SHA256 哈希校验

---

## 架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AI Stroke Painter                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │  输入图片     │───▶│  model/          │───▶│  converter/      │  │
│  │  (jpg/png)   │    │  inference.py    │    │  transform.py    │  │
│  └──────────────┘    └──────────────────┘    └──────────────────┘  │
│                              │                      │              │
│                      ┌───────▼───────┐      ┌───────▼───────┐      │
│                      │ classify_image│      │  deduplicate  │      │
│                      │ (photo/sketch/│      │  _strokes()   │      │
│                      │ illustration) │      │  (可选优化)    │      │
│                      └───────────────┘      └───────────────┘      │
│                              │                      │              │
│                      ┌───────▼───────┐      ┌───────▼───────┐      │
│                      │ RL / Lite     │      │ output_strokes│      │
│                      │ 推理后端      │      │ .json         │      │
│                      └───────────────┘      └───────┬───────┘      │
│                                                     │              │
│                              ┌───────────────────────▼──────────┐  │
│                              │  renderer/                       │  │
│                              │  engine.js  +  index.html        │  │
│                              │  ┌─────────────┐ ┌────────────┐  │  │
│                              │  │renderPainting│ │renderTo    │  │  │
│                              │  │ (逐笔动画)   │ │Offscreen   │  │  │
│                              │  └─────────────┘ └────────────┘  │  │
│                              │  ┌────────────────────────────┐  │  │
│                              │  │ 播放控制面板               │  │  │
│                              │  │ ▶ ⏭ ⏮ 速度 网格 导出 清空  │  │  │
│                              │  └────────────────────────────┘  │  │
│                              └──────────────────────────────────┘  │
│                                                     │              │
│                                              ┌──────▼──────┐       │
│                                              │  Canvas     │       │
│                                              │  逐笔绘画 ✨ │       │
│                                              └─────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

### 工作流程

```
sample.jpg ──► model/inference.py ──► raw_strokes.json
                                           │
                                           ▼
                              converter/transform.py
                                           │
                                           ▼
                                   output_strokes.json
                                           │
                                           ▼
                          renderer/index.html + engine.js
                                           │
                                           ▼
                              Canvas 逐笔绘画回放 ✨
```

---

## 算法原理

### 1. 模型推理（`model/inference.py`）

支持两种后端：

#### RL 模式（Learning-to-Paint）
加载预训练的 Agent（`actor_final.pth`），运行循环策略。Agent 观察当前画布与目标图片的差异，逐步输出笔画动作。神经渲染器模拟画布状态以提供反馈。

**动作向量**（13维）：
```
[x_start, y_start, x_end, y_end,        # 笔画起止点 (0..1)
 color_r, color_g, color_b, color_a,    # 画笔颜色 (0..1)
 brush_radius,                           # 笔刷半径 (0..20px)
 pressure, bend_mid_x, bend_mid_y, stroke_len]  # 辅助参数
```

#### Lite 模式（默认推荐）
纯 NumPy/OpenCV 启发式画家，无需 GPU：

1. **图片类型识别**：基于饱和度+边缘密度+颜色唯一性，区分 photo/sketch/illustration
2. **色彩量化**：K-means 聚类为 8 色调色板
3. **三阶段笔画策略**（根据图片类型调整参数）：
   - **Pass 1 - 色块铺底**：大半径(14px)、低透明度(0.55)，按色彩区域填充
   - **Pass 2 - 中频笔画**：中半径(5px)、中透明度(0.75)，沿区域边缘
   - **Pass 3 - 细节勾画**：细半径(1.8px)、高透明度(0.9)，沿强梯度

| 图片类型 | 策略 | 色块比例 | 中频比例 | 特点 |
|---------|------|---------|---------|------|
| photo | balanced | 45% | 30% | 平衡三阶段 |
| sketch | edge-focused | 0% | 45% | 跳过色块，强调边缘 |
| illustration | colour-focused | 60% | 25% | 增强色块，减少细节 |

### 2. 数据转换（`converter/transform.py`）

核心转换逻辑：

- **颜色映射**：`r,g,b` → `#RRGGBB`，`a` → `["alpha", a]`
- **笔宽映射**：`brush_radius` → `["width", radius]`
- **笔画插值**：`[x0,y0,x1,y1]` 线性插值为 N 个点，N = max(5, 笔长/步长)
- **压感曲线**：钟形分布 `p(t) = 0.15 + 0.85 * 0.5*(1 - cos(2πt))`，起笔收笔压感低
- **笔刷路由**：按 `brush_radius` 自动分配 `brushId`
  - 细线（radius < 3）→ brush 5（压感v3，3值/点：x,y,pressure）
  - 粗线（radius ≥ 3）→ brush 0（马克笔，2值/点：x,y）

**笔画去重优化**（可选，`--dedup`）：
- 签名量化：64×64 网格 + 16 级颜色快速碰撞检测
- 精细校验：位置(±10px)、颜色(±10/255)、半径(±1.5px) 阈值
- 链式合并：连续同色同半径笔画端点延伸

### 3. 渲染引擎（`renderer/engine.js`）

基于原始笔刷引擎源码改造：

- **剥离 Worker 依赖**：移除 `self.addEventListener("message", ...)` 监听
- **暴露 API**：
  - `renderPainting(jsonData, canvas, options)` — 逐笔动画渲染
  - `renderToOffscreen(jsonData, options)` — 离屏预渲染（性能优化）
  - `__paintEngine.applyInstruction(ctx, state, inst, dpr)` — 单指令应用
- **逐笔动画**：通过 `requestAnimationFrame` 每帧绘制 `batch` 条指令
- **进度回调**：`onProgress(done, total)` 实时报告绘制进度

---

## 快速开始

### 环境依赖

- **Python 3.8+**（必需）
- **PyTorch**（可选，仅 RL 模式需要）
- **Node.js 14+**（可选，仅用于本地预览服务器）
- **现代浏览器**（Chrome 90+ / Firefox 88+ / Safari 14+）

### 安装

```bash
git clone https://github.com/huangkunh/AI-Stroke-Painter.git
cd AI-Stroke-Painter

# Python 依赖
pip install opencv-python numpy pillow

# 可选：RL 模式依赖
pip install torch torchvision

# 可选：下载预训练权重（RL 模式）
bash model/download_weights.sh
```

### 运行流程

#### 步骤 1：模型推理

```bash
# Lite 模式（推荐快速演示，无需权重）
python model/inference.py --image assets/sample_cat.jpg --mode lite --max-steps 500

# RL 模式（需要预训练权重）
python model/inference.py --image assets/sample_cat.jpg --mode rl --max-steps 500

# 自动模式（有权重用RL，无则用Lite）
python model/inference.py --image assets/sample_cat.jpg --mode auto --max-steps 500
```

输出：`raw_strokes.json`

#### 步骤 2：数据转换

```bash
# 基本转换
python converter/transform.py --input raw_strokes.json --output output_strokes.json

# 启用笔画去重优化（减少指令数量）
python converter/transform.py --input raw_strokes.json --output output_strokes.json --dedup
```

输出：`output_strokes.json`

#### 步骤 3：浏览器预览

```bash
# 方法1：Python 内置服务器
python -m http.server 8000
# 打开 http://localhost:8000/renderer/index.html

# 方法2：Node.js 服务器
npx serve .
```

在页面中点击「加载 JSON」选择 `output_strokes.json`，然后点击「播放」观看逐笔绘画回放。

---

## API 文档

### Python API

#### `model/inference.py`

```python
from inference import load_image, run_lite_inference, run_rl_inference, classify_image

# 加载图片
image = load_image("sample.jpg", size=512)  # -> np.ndarray (512,512,3) float32

# 图片类型识别
img_type, strategy = classify_image(image)
# img_type: "photo" | "sketch" | "illustration"
# strategy: dict with block_frac, mid_frac, block_radius, etc.

# Lite 推理（无需权重）
actions = run_lite_inference(image, max_steps=500)
# -> List[Dict] with keys: x_start, y_start, x_end, y_end,
#    color_r, color_g, color_b, color_a, brush_radius

# RL 推理（需要权重，自动降级到 Lite）
actions = run_rl_inference(image, "model/pretrained/actor_final.pth", max_steps=500)
```

#### `converter/transform.py`

```python
from transform import transform, to_hex, pick_brush_id, bell_pressure, deduplicate_strokes

# 完整转换
instructions = transform(actions, background="#f8ecdb", start_with_background=True, dedup=False)
# -> List[List] e.g. [["background","#f8ecdb"], ["colour","#ff0000"], ["width",10], ...]

# 辅助函数
hex_str = to_hex(1.0, 0.0, 0.0)        # -> "#ff0000"
brush_id = pick_brush_id(2.5)           # -> 5 (细线)
pressure = bell_pressure(0.5)           # -> ~1.0 (中间压感最高)

# 笔画去重
deduped = deduplicate_strokes(actions)  # -> 去重后的 actions 列表
```

### JavaScript API

#### `renderer/engine.js`

```javascript
// 逐笔动画渲染（返回 Promise）
const result = await renderPainting(jsonData, canvasElement, {
  animate: true,           // 是否动画
  batch: 8,                // 每帧绘制指令数
  background: '#f8ecdb',   // 背景色
  onProgress: (done, total) => console.log(`${done}/${total}`),
  devicePixelRatio: 1,
});
// result: { canvas, strokes }

// 离屏预渲染（同步，用于性能优化）
const off = renderToOffscreen(jsonData, {
  background: '#f8ecdb',
  devicePixelRatio: 1,
  onProgress: (done, total) => {},
});
// off: { canvas, ctx, strokes }
// 用法：ctx.drawImage(off.canvas, 0, 0) 快速 blit

// 单指令应用（用于自定义播放控制）
const state = __paintEngine.cloneState(__paintEngine.defaultState);
__paintEngine.applyInstruction(ctx, state, instruction, 1);
```

#### `renderer/index.html` 播放器 API

```javascript
// 通过 window.__player 访问（浏览器控制台或测试）
window.__player.loadJSON(blob);       // 加载 JSON
window.__player.play();               // 播放
window.__player.pause();              // 暂停
window.__player.step();               // 单步前进
window.__player.reset();              // 重置画布
window.__player.clear();              // 清空画布（卸载JSON）
window.__player.toggleGrid();         // 切换网格显示
window.__player.exportPNG();          // 导出 PNG
window.__player.stepToIndex(500);     // 跳转到指定指令
window.__player.getState();           // 获取状态 {currentIdx, total, isPlaying, showGrid}
```

**键盘快捷键**：
- `Space` — 播放/暂停
- `→` — 单步前进
- `R` — 重置
- `G` — 切换网格
- `C` — 清空画布

---

## 测试

### 运行所有测试

```bash
python -m unittest discover tests -v
```

### 测试覆盖

| 测试文件 | 测试数 | 覆盖内容 |
|---------|--------|---------|
| `tests/test_transform.py` | 38 | `to_hex`、`pick_brush_id`、`bell_pressure`、`interpolate_stroke`、`transform` 全流程 |
| `tests/test_integration.py` | 6 | 图片→推理→转换→JSON 完整管线 |

### 测试详情

**单元测试**（`test_transform.py`）：
- `TestToHex`：黑/白/红/绿/蓝/中间值/越界钳制/格式校验
- `TestPickBrushId`：细线→brush5, 粗线→brush0, 边界值
- `TestBellPressure`：起笔收笔压感 < 中间压感（核心要求）、对称性、单调性、值域
- `TestInterpolateStroke`：端点匹配、点数、线性插值
- `TestTransformPipeline`：背景指令、颜色/宽度指令、笔刷路由、坐标缩放

**集成测试**（`test_integration.py`）：
- 合成图像完整管线
- 真实示例图像（sample_landscape, sample_sketch）
- 笔画计数、颜色范围、坐标范围
- JSON 格式有效性
- Lite 模式确定性

---

## 部署

详细部署指南请参见 [docs/deployment.md](docs/deployment.md)。

### 快速部署

#### Docker

```bash
docker build -t ai-stroke-painter .
docker run -p 8000:8000 ai-stroke-painter
# 打开 http://localhost:8000
```

#### 本地

```bash
pip install opencv-python numpy pillow
python model/inference.py --image assets/sample_cat.jpg --mode lite
python converter/transform.py --input raw_strokes.json --output output_strokes.json
python -m http.server 8000
```

---

## 教程

- [基础使用教程](docs/tutorials/basic.md) — 从零开始完成第一次绘画
- [高级功能教程](docs/tutorials/advanced.md) — 图片类型识别、去重、Worker池、虚拟滚动
- [故障排除指南](docs/tutorials/troubleshooting.md) — 常见问题与解决方案
- [REST API 指南](docs/api/rest.md) — 云函数 REST API 完整文档
- [Python API 指南](docs/api/python.md) — Python API 使用指南
- [JavaScript API 指南](docs/api/javascript.md) — JavaScript API 使用指南
- [部署指南](docs/deployment.md) — 本地/Docker/云平台部署

### 示例图片库

`assets/samples/` 目录包含多种风格的测试图片：

| 图片 | 风格 | 识别类型 |
|------|------|----------|
| `sample_portrait.png` | 人像 | photo |
| `sample_mountains.png` | 风景 | illustration |
| `sample_abstract.png` | 抽象 | illustration |
| `sample_anime.png` | 动漫 | illustration |
| `sample_bird.png` | 动物 | illustration |
| `sample_flower.png` | 花卉 | illustration |
| `sample_geometric.png` | 几何 | illustration |
| `sample_underwater.png` | 水下 | illustration |
| `sample_cat_lowres.png` | 低分辨率猫图 | photo |

---

## FAQ

### Q1: RL 模式和 Lite 模式有什么区别？

**RL 模式**使用 Learning-to-Paint 预训练的神经网络 Agent，笔画质量更高但需要下载权重（~MB级）和安装 PyTorch。**Lite 模式**是纯 NumPy/OpenCV 的启发式画家，无需任何依赖，适合快速演示。当 RL 权重不可用时，系统会自动降级到 Lite 模式。

### Q2: 为什么我的图片识别为错误的类型？

图片类型识别基于饱和度、边缘密度和颜色唯一性三个启发式指标。如果识别不准确，可以手动指定策略：修改 `model/inference.py` 中 `classify_image()` 的阈值，或在 `run_lite_inference()` 中直接传入自定义 strategy dict。

### Q3: `--dedup` 选项会影响绘画质量吗？

`--dedup` 会移除位置、颜色、半径都极其相似的重复笔画，以及合并端点相连的同色笔画。这会减少指令数量（加速渲染），视觉上几乎无差异。如果需要精确还原，不要使用此选项。

### Q4: 渲染引擎支持哪些笔刷？

引擎内置 70+ 种笔刷（brush 0-72）。本项目主要使用两种：
- **brush 0（马克笔）**：平涂色块，2值/点 (x,y)，用于粗线（radius≥3）
- **brush 5（压感v3）**：压感细线，3值/点 (x,y,pressure)，用于细线（radius<3）

### Q5: 如何导出绘画过程为视频？

目前支持导出当前画布为 PNG。如需视频，可以：
1. 使用浏览器的 MediaRecorder API 录制 Canvas
2. 定期截图（`canvas.toDataURL()`）然后用 ffmpeg 合成视频

### Q6: 权重下载失败怎么办？

Google Drive 有时会有配额限制。可以：
1. 手动从 [ICCV2019-LearningToPaint](https://github.com/hzwer/ICCV2019-LearningToPaint) 下载 `actor_final.pth`
2. 放到 `model/pretrained/actor_final.pth`
3. 运行 `bash model/download_weights.sh --check` 验证完整性
4. 或直接使用 `--mode lite` 无需权重

### Q7: 支持哪些图片格式？

输入支持 JPG、PNG、BMP、WEBP 等所有 PIL 支持的格式。输出 JSON 为标准 UTF-8 编码。建议输入图片尺寸 ≥ 256×256 以获得最佳效果。

---

## 贡献

欢迎贡献！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解代码风格、提交规范和测试要求。

---

## 致谢

- [Learning-to-Paint (ICCV 2019)](https://github.com/hzwer/ICCV2019-LearningToPaint) — Zheng et al. 的模型基础与预训练权重
- 原始笔刷渲染引擎 — 提供了压感笔刷与马克笔的 Canvas 2D 实现

---

## License

MIT
