# AI Stroke Painter

> 将输入图片转化为具有人类绘画时序的笔画数据，并通过自定义 Canvas 2D 引擎逐笔回放渲染。

本项目基于强化学习模型 [Learning-to-Paint](https://github.com/hzwer/ICCV2019-LearningToPaint)（ICCV 2019），实现了一条完整的 **图片 → AI 笔画推理 → 数据转换 → 浏览器逐笔回放** 管线。

## 项目简介

AI Stroke Painter 分为三层：

| 层 | 目录 | 职责 |
|---|---|---|
| **模型推理端** | `model/` | 加载预训练 Agent，对目标图片运行强化学习推理循环，输出每一笔的动作参数 |
| **数据转换层** | `converter/` | 将模型输出的原始动作张量转换为渲染引擎可识别的扁平 JSON 指令流 |
| **渲染引擎端** | `renderer/` | 封装自定义 Canvas 2D 笔刷引擎，接收 JSON 指令并在画布上逐笔绘制 |

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

### 模型动作参数

模型每次推理输出一个 13 维动作向量，其中前 9 维为核心笔画参数：

| 索引 | 参数 | 范围 | 说明 |
|------|------|------|------|
| 0-1 | x_start, y_start | 0~1 | 笔画起点（归一化坐标） |
| 2-3 | x_end, y_end | 0~1 | 笔画终点 |
| 4-6 | color_r/g/b | 0~1 | 画笔颜色 |
| 7 | color_a | 0~1 | 透明度 |
| 8 | brush_radius | 0~20px | 笔刷半径 |

### 渲染引擎指令格式

转换层将动作参数映射为以下扁平指令流：

```json
[
  ["background", "#f8ecdb"],
  ["colour", "#d20000"],
  ["width", 6],
  ["alpha", 0.8],
  ["line", 0, 100, 100, 4, 150, 120, 8, 200, 100, 4],
  ...
]
```

| 指令 | 格式 | 说明 |
|------|------|------|
| `background` | `["background", "#RRGGBB"]` | 设置背景色 |
| `colour` | `["colour", "#RRGGBB"]` | 设置画笔颜色 |
| `width` | `["width", N]` | 设置基础笔宽 |
| `alpha` | `["alpha", 0~1]` | 设置透明度 |
| `line` | `["line", brushId, x1,y1,p1, x2,y2,p2, ...]` | 绘制笔画，点数据扁平化 |

**笔刷选择规则**：
- `brush_radius < 3` → `brushId = 5`（压感v3，适合勾线）
- `brush_radius >= 3` → `brushId = 0`（马克笔，适合平涂铺色）

**压感插值**：每条笔画在起点和终点之间线性插值 5~10 个中间点，压感值呈钟形分布（起笔收笔低、中间高），模拟人类手绘节奏。

## 环境依赖

### Python 端（模型推理 + 数据转换）

```
Python >= 3.8
PyTorch >= 1.10      # 仅 RL 模式需要，lite 模式无需
OpenCV-Python (cv2)
Pillow (PIL)
NumPy
gdown                # 下载预训练权重
```

安装：
```bash
pip install torch torchvision opencv-python pillow numpy gdown
```

### 前端（渲染回放）

无需安装依赖，纯静态 HTML + JS。只需一个现代浏览器（Chrome / Firefox / Edge / Safari）。

如需本地服务器预览：
```bash
# 方式一：Python 内置
python -m http.server 8080 -d renderer

# 方式二：Node.js
npx serve renderer
```

## 运行流程

### 步骤 1：模型推理

```bash
# lite 模式（无需 GPU / PyTorch，使用启发式画家，推荐快速体验）
python model/inference.py --image sample.jpg --mode lite --max-steps 500

# RL 模式（需先下载预训练权重）
bash model/download_weights.sh
python model/inference.py --image sample.jpg --mode rl --max-steps 600
```

输出：`raw_strokes.json`（包含每笔的动作参数）

### 步骤 2：数据转换

```bash
python converter/transform.py --input raw_strokes.json --output output_strokes.json
```

输出：`output_strokes.json`（渲染引擎可识别的扁平指令流）

### 步骤 3：浏览器逐笔回放

```bash
# 启动本地服务器
python -m http.server 8080 -d renderer
```

浏览器打开 `http://localhost:8080`，点击「加载 output_strokes.json」上传文件，点击「播放」观看逐笔绘画回放。

## 目录结构

```
AI-Stroke-Painter/
├── model/
│   ├── network.py            # Learning-to-Paint 模型架构（Agent + Renderer）
│   ├── inference.py          # 推理入口（RL 模式 + lite 启发式模式）
│   └── download_weights.sh   # 预训练权重下载脚本
├── converter/
│   └── transform.py          # 核心转换层：动作参数 → 引擎 JSON
├── renderer/
│   ├── engine.js             # 封装后的 Canvas 2D 笔刷渲染引擎
│   └── index.html            # 前端预览界面（文件上传 + 逐笔动画）
├── assets/                   # 示例图片与演示截图
├── .gitignore
└── README.md
```

## 技术细节

### 模型推理（`model/inference.py`）

支持两种后端：

1. **RL 模式**：加载 Learning-to-Paint 预训练的 Agent（`actor_final.pth`），运行循环策略。Agent 观察当前画布与目标图片的差异，逐步输出笔画动作。神经渲染器模拟画布状态以提供反馈。

2. **Lite 模式**（默认推荐）：纯 NumPy/OpenCV 启发式画家，无需 GPU：
   - 色彩量化为有限调色板
   - 先铺大色块（粗笔、低透明度）
   - 再补中频笔画（沿区域边缘）
   - 最后勾细节（细笔、沿强梯度）
   
   产出视觉上接近人类绘画顺序的笔画序列，适合快速演示。

### 数据转换（`converter/transform.py`）

核心转换逻辑：

- **颜色映射**：`r,g,b` → `#RRGGBB`，`a` → `["alpha", a]`
- **笔宽映射**：`brush_radius` → `["width", radius]`
- **笔画插值**：`[x0,y0,x1,y1]` 线性插值为 N 个点，N = max(5, 笔长/步长)
- **压感曲线**：钟形分布 `p(t) = base + peak * sin(πt)`，起笔收笔压感低
- **笔刷路由**：按 `brush_radius` 自动分配 `brushId`（细线→5，粗线→0）

### 渲染引擎（`renderer/engine.js`）

基于原始笔刷引擎源码改造：

- **剥离 Worker 依赖**：移除 `self.addEventListener("message", ...)` 监听
- **暴露 API**：`renderPainting(jsonData, canvasElement, options)` 异步函数
- **逐笔动画**：通过 `requestAnimationFrame` 每帧绘制 `batch` 条指令
- **进度回调**：`onProgress(done, total)` 实时报告绘制进度

## 致谢

- [Learning-to-Paint (ICCV 2019)](https://github.com/hzwer/ICCV2019-LearningToPaint) — Zheng et al. 的模型基础与预训练权重
- 原始笔刷渲染引擎 — 提供了压感笔刷与马克笔的 Canvas 2D 实现

## License

MIT
