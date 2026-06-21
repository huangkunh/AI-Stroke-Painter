# 教程视频脚本

本文件包含 AI Stroke Painter 系列教程视频的脚本，展示完整使用流程。

---

## 视频 1: 基础使用 — 三步完成图片转绘画 (5分钟)

### 开场 (0:00-0:30)

**画面**: 项目首页截图 + 标题 "AI Stroke Painter - 三步完成图片转绘画"

**旁白**:
> 大家好，今天我来演示如何使用 AI Stroke Painter，将一张普通图片转化为逐笔绘画的动画。整个过程只需要三步。

### 步骤 1: 模型推理 (0:30-2:00)

**画面**: 终端窗口，输入命令

**旁白**:
> 第一步，运行模型推理。我们使用 Lite 模式，它不需要 GPU，纯 Python 即可运行。
>
> ```bash
> python model/inference.py --image assets/sample_cat.jpg --mode lite --max-steps 500
> ```
>
> `--max-steps` 控制笔画数量，500 笔适合大多数图片。系统会自动识别图片类型，这里识别为 photo，使用平衡策略。
>
> 推理完成后，生成 `raw_strokes.json`，包含每一笔的起点、终点、颜色和半径。

**画面**: 显示 raw_strokes.json 的内容片段

### 步骤 2: 数据转换 (2:00-3:00)

**画面**: 终端窗口

**旁白**:
> 第二步，将原始笔画数据转换为渲染引擎能识别的 JSON 格式。
>
> ```bash
> python converter/transform.py --input raw_strokes.json --output output_strokes.json --dedup
> ```
>
> `--dedup` 选项启用笔画去重，减少重复笔画，加速渲染。这里 500 笔去重后变成 480 笔。
>
> 转换器会自动根据笔画半径选择笔刷：细线用压感笔刷，粗线用马克笔。

### 步骤 3: 浏览器预览 (3:00-4:30)

**画面**: 启动服务器 + 打开浏览器

**旁白**:
> 第三步，启动本地服务器，在浏览器中观看逐笔绘画。
>
> ```bash
> python -m http.server 8000
> ```
>
> 打开 `http://localhost:8000/renderer/index.html`，点击「加载 JSON」选择 `output_strokes.json`。
>
> 点击播放按钮，可以看到画笔一笔一笔地绘制。状态栏实时显示进度和帧率。
>
> 可以用速度下拉框调整播放速度，单步按钮逐笔查看，网格按钮显示参考网格。

### 结尾 (4:30-5:00)

**画面**: 完成的绘画作品

**旁白**:
> 最后，点击「导出 PNG」保存当前画布。整个流程就完成了！
>
> 下一期视频，我会介绍高级功能，包括图片类型识别、笔画去重和性能优化。敬请关注！

---

## 视频 2: 高级功能 — 图片类型识别与性能优化 (8分钟)

### 开场 (0:00-0:30)

**画面**: 三种图片类型对比

**旁白**:
> 本期视频介绍 AI Stroke Painter 的高级功能。系统会自动识别图片类型，针对不同类型采用不同的笔画策略。

### 图片类型识别 (0:30-3:00)

**画面**: 三张示例图片 — 照片、线稿、插画

**旁白**:
> 系统基于饱和度和边缘密度，将图片分为三类：
>
> 1. **照片 (photo)**: 自然色彩分布，使用平衡的三阶段策略 — 先铺大色块，再补中频笔画，最后勾细节。
>
> 2. **线稿 (sketch)**: 低饱和度、高边缘密度，跳过色块铺底，直接用细笔强调边缘。
>
> 3. **插画 (illustration)**: 高饱和度、少颜色，增强色块铺底，减少细节笔画。
>
> 让我们用三张图片测试：

**画面**: 运行三张图片的推理

```bash
python model/inference.py --image assets/sample_cat.jpg --mode lite    # photo
python model/inference.py --image assets/sample_sketch.png --mode lite # sketch
python model/inference.py --image assets/sample_landscape.png --mode lite # illustration
```

**旁白**:
> 可以看到，猫图识别为 photo，简笔画识别为 sketch（pass 1 色块铺底为 0 笔），风景画识别为 illustration。

### 笔画去重 (3:00-5:00)

**画面**: 对比有去重和无去重的输出

**旁白**:
> `--dedup` 选项启用笔画去重和链式合并：
>
> - **精确去重**: 位置、颜色、半径都相似的笔画被移除
> - **链式合并**: 端点相连的同色笔画合并为一条
>
> 通常减少 5-15% 的笔画数量，视觉上几乎无差异，但渲染速度更快。

### 性能优化 (5:00-7:30)

**画面**: 播放器界面，显示 FPS 和 batch size

**旁白**:
> 播放器集成了自适应帧率控制（AFC），根据设备性能动态调整渲染批次大小。
>
> 状态栏显示 `~60fps batch=64`，表示当前约 60 帧/秒，每帧绘制 64 条指令。
>
> 对于超长笔画序列（10k+），引擎提供虚拟滚动渲染器，通过检查点机制实现快速跳转。

### 结尾 (7:30-8:00)

**画面**: 项目 GitHub 页面

**旁白**:
> 更多功能请查看项目文档。下一期，我会介绍 REST API 和云部署。感谢观看！

---

## 视频 3: REST API 与云部署 (6分钟)

### 开场 (0:00-0:30)

**旁白**:
> 本期介绍如何通过 REST API 调用 AI Stroke Painter，以及部署到 Vercel 云平台。

### 本地 API 测试 (0:30-3:00)

**画面**: 终端运行 API CLI 模式

**旁白**:
> 三个 API 端点都支持 CLI 模式本地测试：
>
> ```bash
> # 推理
> python api/infer.py --image input.jpg --mode lite --max-steps 400
>
> # 转换
> python api/transform.py --input strokes.json --dedup
>
> # 组合管线（一次调用完成推理+转换）
> python api/pipeline.py --image input.jpg --mode lite --dedup
> ```

### Vercel 部署 (3:00-5:00)

**画面**: Vercel 部署界面

**旁白**:
> 部署到 Vercel 只需要三步：
>
> 1. Fork 项目到你的 GitHub
> 2. 在 Vercel 导入项目
> 3. 自动部署，前端 + API 同时上线
>
> `vercel.json` 已配置多区域部署（美西/美东/东京/新加坡/法兰克福），CDN 缓存，和函数资源限制。

### API 调用示例 (5:00-6:00)

**画面**: Python 代码调用 API

**旁白**:
> 部署后，用 Python 调用：
>
> ```python
> import requests, base64
>
> with open('input.jpg', 'rb') as f:
>     img_b64 = base64.b64encode(f.read()).decode()
>
> resp = requests.post('https://your-app.vercel.app/api/pipeline', json={
>     'image': img_b64, 'mode': 'lite', 'max_steps': 400, 'dedup': True
> })
> instructions = resp.json()['instructions']
> ```

### 结尾 (6:00)

**旁白**:
> 感谢观看完整教程系列！如有问题请查看故障排除指南或提交 Issue。

---

## 拍摄建议

1. **分辨率**: 1920x1080，30fps
2. **字体**: 终端用等宽字体（如 Fira Code），16-18px
3. **高亮**: 关键命令用黄色高亮
4. **节奏**: 每个步骤留 2-3 秒让观众看清
5. **字幕**: 添加中英文字幕，方便国际观众
