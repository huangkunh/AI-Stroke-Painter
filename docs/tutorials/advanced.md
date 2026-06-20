# 高级功能教程

本教程介绍 AI Stroke Painter 的高级功能。

## 1. 图片类型识别

系统自动识别图片类型并调整笔画策略：

| 类型 | 识别特征 | 笔画策略 |
|------|----------|----------|
| photo | 自然色彩分布 | 平衡 3-pass（色块+中频+细节） |
| sketch | 低饱和度+高边缘密度 | 跳过色块铺底，强调边缘笔画 |
| illustration | 高饱和度+少颜色 | 增强色块铺底，减少细节 |

### 手动指定策略

```python
from model.inference import classify_image, run_lite_inference, load_image

image = load_image("input.jpg", size=512)
img_type, strategy = classify_image(image)
print(f"Detected: {img_type}, strategy: {strategy['name']}")

# 手动覆盖策略
custom_strategy = {
    "name": "custom",
    "block_frac": 0.5, "block_radius": 12.0, "block_alpha": 0.6,
    "mid_frac": 0.3, "mid_radius": 4.0, "mid_alpha": 0.8,
    "fine_radius": 1.5, "fine_alpha": 0.9,
}
# 直接调用内部函数...
```

## 2. 笔画去重

`--dedup` 选项启用笔画去重和链式合并：

```bash
python converter/transform.py --input raw.json --output out.json --dedup
```

**去重策略**：
1. **精确去重**：位置、颜色、半径都相似的笔画被移除
2. **链式合并**：端点相连的同色笔画合并为一条

**效果**：通常减少 5-15% 的笔画数量，视觉上几乎无差异。

## 3. 离屏预渲染

对于超长笔画序列（10k+），使用离屏预渲染加速：

```javascript
// 一次性渲染全部指令到离屏 canvas
const offscreen = globalThis.renderToOffscreen(instructions, {
  background: '#f8ecdb',
  devicePixelRatio: 1
});

// 用 drawImage 快速 blit 到主 canvas
ctx.drawImage(offscreen.canvas, 0, 0);
```

## 4. 虚拟滚动渲染

对于 10k+ 笔画的超长序列，使用虚拟滚动：

```javascript
const vsr = globalThis.createVirtualStrokeRenderer(canvas, {
  background: '#f8ecdb',
  bufferSize: 500  // 每 500 笔创建一个检查点
});

vsr.load(instructions);
vsr.seekTo(5000);    // 跳转到第 5000 笔
vsr.forward(100);    // 前进 100 笔
console.log(vsr.getCheckpointCount());  // 检查点数量
```

## 5. Worker 池并行预处理

使用 Web Worker 并行处理笔画平滑/重采样：

```javascript
const pool = globalThis.createStrokeProcessorPool({ size: 4 });

// 批量预处理
pool.preprocess(instructions, { segments: 8 }).then(map => {
  console.log('Processed', Object.keys(map).length, 'strokes');
});

// 单条平滑
pool.smooth([0,0, 10,10, 20,5], 8).then(smoothed => {
  console.log('Smoothed:', smoothed);
});

// 用完记得终止
pool.terminate();
```

## 6. 自适应帧率

根据设备性能动态调整渲染帧率：

```javascript
const afc = globalThis.createAdaptiveFrameRate({ targetFps: 60 });

function tick(ts) {
  const start = performance.now();
  // ... 渲染逻辑 ...
  const duration = performance.now() - start;
  afc.recordFrame(duration);
  
  const batch = afc.suggestBatchSize(8);
  console.log(afc.getStats());  // { avgFrameMs, currentBatch, ... }
  
  requestAnimationFrame(tick);
}
```

## 7. REST API

部署云函数后，可通过 REST API 调用：

```bash
# 推理
curl -X POST https://your-app.vercel.app/api/infer \
  -H "Content-Type: application/json" \
  -d '{"image":"<base64>","mode":"lite","max_steps":400}'

# 转换
curl -X POST https://your-app.vercel.app/api/transform \
  -H "Content-Type: application/json" \
  -d '{"strokes":[...],"dedup":true}'
```

详见 [API 文档](../api.md)。

## 8. 自定义快捷键

在前端界面点击「快捷键」按钮自定义：

- 播放/暂停
- 单步前进
- 重置
- 网格切换
- 清空画布
- 导出 PNG

配置保存在 `localStorage`，下次访问自动加载。

## 9. 笔画高亮与导航

- **悬停高亮**：鼠标悬停在画布上，高亮显示对应笔画
- **信息提示**：显示笔画序号、颜色、半径
- **缩略图导航**：点击「导航侧栏」打开缩略图列表，点击跳转

## 10. 触摸手势

移动端支持：
- **双指缩放**：放大/缩小画布
- **滑动导航**：左右滑动前进/后退笔画
