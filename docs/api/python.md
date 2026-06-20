# Python API 使用指南

本指南介绍如何通过 Python API 使用 AI Stroke Painter 的核心功能。

## 目录

- [安装](#安装)
- [模型推理 API](#模型推理-api)
- [数据转换 API](#数据转换-api)
- [图片类型识别 API](#图片类型识别-api)
- [完整示例](#完整示例)

---

## 安装

```bash
pip install opencv-python numpy pillow
# 可选（RL模式）:
pip install torch torchvision
```

## 模型推理 API

### `model.inference.load_image(path, size=512)`

加载图片并缩放到指定尺寸。

**参数**:
- `path` (str): 图片路径
- `size` (int): 目标尺寸（正方形），默认 512

**返回**: `np.ndarray` — 形状 `(size, size, 3)` 的 float32 数组，值域 [0, 1]

**示例**:
```python
import sys
sys.path.insert(0, 'model')
from inference import load_image

image = load_image('assets/sample_cat.jpg', size=512)
print(image.shape)  # (512, 512, 3)
```

### `model.inference.run_lite_inference(image, max_steps=600)`

运行 Lite 模式推理（纯 NumPy/OpenCV，无需 PyTorch）。

**参数**:
- `image` (np.ndarray): 输入图片，形状 `(H, W, 3)`，值域 [0, 1]
- `max_steps` (int): 最大笔画数，默认 600

**返回**: `List[Dict]` — 笔画动作字典列表

**示例**:
```python
from inference import load_image, run_lite_inference

image = load_image('input.jpg', size=512)
strokes = run_lite_inference(image, max_steps=400)
print(f"Generated {len(strokes)} strokes")
print(strokes[0])
# {'x_start': 0.5, 'y_start': 0.3, 'x_end': 0.6, 'y_end': 0.4,
#  'color_r': 0.8, 'color_g': 0.2, 'color_b': 0.1, 'color_a': 0.7,
#  'brush_radius': 12.0}
```

### `model.inference.run_rl_inference(image, weights_path, max_steps, device='cpu')`

运行 RL 模式推理（需要 PyTorch + 预训练权重）。

**参数**:
- `image` (np.ndarray): 输入图片
- `weights_path` (str): 权重文件路径
- `max_steps` (int): 最大笔画数
- `device` (str): 计算设备，默认 'cpu'

**返回**: `List[Dict]` — 笔画动作字典列表

**自动降级**: 如果权重文件不存在或加载失败，自动降级到 Lite 模式。

```python
from inference import load_image, run_rl_inference

image = load_image('input.jpg', size=512)
strokes = run_rl_inference(
    image,
    weights_path='model/pretrained/actor_final.pth',
    max_steps=400,
    device='cpu'
)
```

## 数据转换 API

### `converter.transform.transform(actions, background='#f8ecdb', start_with_background=True, dedup=False)`

将原始笔画数据转换为渲染引擎 JSON 指令流。

**参数**:
- `actions` (List[Dict]): 笔画动作字典列表
- `background` (str): 背景色，默认 '#f8ecdb'
- `start_with_background` (bool): 是否输出背景指令，默认 True
- `dedup` (bool): 是否启用笔画去重，默认 False

**返回**: `List[List]` — 引擎指令列表

**示例**:
```python
from converter.transform import transform

instructions = transform(
    strokes,
    background='#f8ecdb',
    dedup=True  # 启用去重减少指令数量
)
print(f"Generated {len(instructions)} instructions")
print(instructions[0])  # ['background', '#f8ecdb']
print(instructions[1])  # ['colour', '#a18256']
```

### `converter.transform.deduplicate_strokes(actions)`

笔画去重和链式合并。

**参数**:
- `actions` (List[Dict]): 笔画动作列表

**返回**: `List[Dict]` — 去重后的笔画列表

```python
from converter.transform import deduplicate_strokes

deduped = deduplicate_strokes(strokes)
print(f"Before: {len(strokes)}, After: {len(deduped)}")
```

## 图片类型识别 API

### `model.inference.classify_image(image)`

识别图片类型（photo/sketch/illustration）。

**参数**:
- `image` (np.ndarray): 输入图片

**返回**: `tuple` — `(type_name, strategy_dict)`

```python
from inference import load_image, classify_image

image = load_image('input.jpg', size=256)
img_type, strategy = classify_image(image)
print(f"Type: {img_type}")        # 'photo' / 'sketch' / 'illustration'
print(f"Strategy: {strategy['name']}")  # 'balanced' / 'edge-focused' / 'colour-focused'
```

## 完整示例

```python
import sys
import json
sys.path.insert(0, 'model')
sys.path.insert(0, 'converter')

from inference import load_image, run_lite_inference, classify_image
from transform import transform

# 1. 加载图片
image = load_image('assets/sample_cat.jpg', size=512)

# 2. 识别图片类型
img_type, strategy = classify_image(image)
print(f"Image type: {img_type}, strategy: {strategy['name']}")

# 3. 运行推理
strokes = run_lite_inference(image, max_steps=400)
print(f"Generated {len(strokes)} strokes")

# 4. 转换为引擎指令
instructions = transform(strokes, background='#f8ecdb', dedup=True)
print(f"Generated {len(instructions)} instructions")

# 5. 保存
with open('output_strokes.json', 'w') as f:
    json.dump(instructions, f)
print("Saved to output_strokes.json")
```
