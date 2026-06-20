# JavaScript API 使用指南

本指南介绍如何通过 JavaScript API 使用 AI Stroke Painter 的渲染引擎和播放器。

## 目录

- [渲染引擎 API](#渲染引擎-api)
- [播放器 API](#播放器-api)
- [增强功能 API](#增强功能-api)
- [完整示例](#完整示例)

---

## 渲染引擎 API

### `globalThis.renderPainting(jsonData, canvasElement, options)`

将 JSON 指令流渲染到 Canvas（异步，支持动画）。

**参数**:
- `jsonData` (Array): 引擎指令数组
- `canvasElement` (HTMLCanvasElement): 目标 canvas 元素
- `options` (Object):
  - `animate` (bool): 是否动画播放，默认 true
  - `batch` (number): 每帧绘制的指令数，默认 8
  - `background` (string): 背景色
  - `onProgress` (function): 进度回调 `(done, total) => void`
  - `devicePixelRatio` (number): 设备像素比，默认 1

**返回**: `Promise<{canvas, strokes}>`

**示例**:
```javascript
const canvas = document.getElementById('canvas');
const jsonData = [["background", "#f8ecdb"], ["colour", "#ff0000"], ...];

renderPainting(jsonData, canvas, {
  animate: true,
  batch: 8,
  background: '#f8ecdb',
  onProgress: (done, total) => {
    console.log(`${done}/${total} (${(done/total*100).toFixed(1)}%)`);
  }
}).then(result => {
  console.log(`Done: ${result.strokes} strokes`);
});
```

### `globalThis.renderToOffscreen(jsonData, options)`

将全部指令一次性渲染到离屏 Canvas（同步，用于预渲染加速）。

**参数**:
- `jsonData` (Array): 引擎指令数组
- `options` (Object):
  - `background` (string): 背景色
  - `devicePixelRatio` (number): 设备像素比
  - `onProgress` (function): 进度回调

**返回**: `{canvas, ctx, strokes}`

```javascript
const off = renderToOffscreen(jsonData, { background: '#f8ecdb' });
// 用 drawImage 快速 blit 到主 canvas
ctx.drawImage(off.canvas, 0, 0);
```

### `globalThis.createStrokeProcessorPool(opts)`

创建 Worker 池用于并行笔画预处理。

**参数**:
- `opts` (Object):
  - `size` (number): Worker 数量，默认 `min(4, hardwareConcurrency)`
  - `workerUrl` (string): Worker 脚本路径，默认 'stroke-processor.js'

**返回**: Worker 池对象

```javascript
const pool = createStrokeProcessorPool({ size: 4 });

// 并行预处理
pool.preprocessParallel(instructions, { segments: 8 })
  .then(map => {
    console.log(`Processed ${Object.keys(map).length} strokes`);
  });

// 单条平滑
pool.smooth([0,0, 10,10, 20,5], 8).then(smoothed => {
  console.log(smoothed);
});

// 用完记得终止
pool.terminate();
```

### `globalThis.createAdaptiveFrameRate(opts)`

创建自适应帧率控制器。

**参数**:
- `opts` (Object):
  - `targetFps` (number): 目标帧率，默认 60
  - `minBatch` (number): 最小批量，默认 1
  - `maxBatch` (number): 最大批量，默认 64
  - `window` (number): 滚动窗口大小，默认 10
  - `initialBatch` (number): 初始批量，默认 8

```javascript
const afc = createAdaptiveFrameRate({ targetFps: 60 });

function tick(ts) {
  const start = performance.now();
  // ... 渲染逻辑 ...
  const duration = performance.now() - start;
  afc.recordFrame(duration);

  const batch = afc.suggestBatchSize(8);
  console.log(afc.getStats());
  // { avgFrameMs, currentBatch, targetFps, historyLen }

  requestAnimationFrame(tick);
}
```

### `globalThis.createVirtualStrokeRenderer(canvasElement, opts)`

创建虚拟滚动渲染器（支持超长笔画序列）。

**参数**:
- `canvasElement` (HTMLCanvasElement): 目标 canvas
- `opts` (Object):
  - `background` (string): 背景色
  - `bufferSize` (number): 检查点间隔，默认 500
  - `devicePixelRatio` (number): 设备像素比

```javascript
const vsr = createVirtualStrokeRenderer(canvas, {
  background: '#f8ecdb',
  bufferSize: 500
});

vsr.load(instructions);
vsr.seekTo(5000);    // 跳转到第 5000 笔
vsr.forward(100);    // 前进 100 笔
vsr.seekTo(30);      // 后退到第 30 笔

console.log(vsr.getCheckpointCount());  // 检查点数量
```

## 播放器 API

### `window.__player`

主播放器 API，在 `index.html` 中暴露。

**方法**:
- `loadJSON(file)`: 加载 JSON 文件
- `play()`: 开始播放
- `pause()`: 暂停播放
- `step()`: 单步前进
- `reset()`: 重置到开头
- `clear()`: 清空画布
- `toggleGrid()`: 切换网格
- `exportPNG()`: 导出 PNG
- `stepToIndex(idx)`: 跳转到指定指令
- `getInstructions()`: 获取当前指令数组
- `getState()`: 获取播放器状态

```javascript
// 获取状态
const state = window.__player.getState();
console.log(state);
// { currentIdx: 100, total: 2001, isPlaying: false, showGrid: false }

// 跳转到指定位置
window.__player.stepToIndex(500);

// 导出 PNG
window.__player.exportPNG();
```

## 增强功能 API

### `window.__playerEnh`

增强功能 API，在 `player-enhancements.js` 中暴露。

**方法**:
- `buildThumbnails()`: 构建缩略图导航
- `shortcutsPanel()`: 打开快捷键设置面板
- `getShortcuts()`: 获取当前快捷键配置
- `resetShortcuts()`: 重置快捷键到默认值

```javascript
// 构建缩略图导航
window.__playerEnh.buildThumbnails();

// 获取快捷键
const shortcuts = window.__playerEnh.getShortcuts();
console.log(shortcuts);
// { playPause: 'Space', stepForward: 'ArrowRight', reset: 'KeyR', ... }

// 重置快捷键
window.__playerEnh.resetShortcuts();
```

## 完整示例

```html
<!DOCTYPE html>
<html>
<head>
  <script src="engine.js"></script>
  <script src="player-enhancements.js"></script>
</head>
<body>
  <canvas id="canvas" width="640" height="480"></canvas>
  <script>
    // 加载 JSON 并播放
    fetch('output_strokes.json')
      .then(r => r.json())
      .then(jsonData => {
        const canvas = document.getElementById('canvas');
        return renderPainting(jsonData, canvas, {
          animate: true,
          batch: 8,
          background: '#f8ecdb',
          onProgress: (done, total) => {
            console.log(`${done}/${total}`);
          }
        });
      })
      .then(result => {
        console.log(`Done: ${result.strokes} strokes`);
      });
  </script>
</body>
</html>
```
