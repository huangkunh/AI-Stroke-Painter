# 贡献指南

感谢你对 AI Stroke Painter 项目的关注！本文档明确了代码风格、提交规范和测试要求。

---

## 目录

- [开发环境搭建](#开发环境搭建)
- [代码风格](#代码风格)
- [提交规范](#提交规范)
- [测试要求](#测试要求)
- [Pull Request 流程](#pull-request-流程)
- [项目结构](#项目结构)

---

## 开发环境搭建

### 前置条件

- Python 3.8+
- Node.js 14+（用于前端预览）
- Git

### 步骤

```bash
# 1. Fork 并克隆仓库
git clone https://github.com/your-username/AI-Stroke-Painter.git
cd AI-Stroke-Painter

# 2. 添加上游远程
git remote add upstream https://github.com/huangkunh/AI-Stroke-Painter.git

# 3. 创建开发分支
git checkout -b feat/your-feature-name

# 4. 安装 Python 依赖
pip install opencv-python numpy pillow
pip install torch torchvision  # 可选，用于 RL 模式

# 5. 运行测试确保环境正常
python -m unittest discover tests
```

---

## 代码风格

### Python 代码

遵循 [PEP 8](https://peps.python.org/pep-0008/) 规范。

**关键要求**：

- 缩进：4 个空格
- 行宽：不超过 100 字符
- 导入顺序：标准库 → 第三方库 → 本地模块
- 函数/类文档字符串：使用三引号，描述参数和返回值
- 类型注解：公共函数应添加类型注解

**示例**：

```python
#!/usr/bin/env python3
"""Module docstring."""

from __future__ import annotations

import os
import sys
from typing import List, Dict

import numpy as np


def transform_strokes(
    actions: List[Dict[str, float]],
    dedup: bool = False
) -> List[List]:
    """Convert raw strokes to engine instructions.

    Args:
        actions: List of stroke action dicts.
        dedup: If True, enable stroke deduplication.

    Returns:
        List of engine instruction tuples.
    """
    if dedup:
        actions = deduplicate_strokes(actions)
    # ... implementation
```

**工具**：

```bash
# 检查代码风格
pip install flake8
flake8 model/ converter/ tests/

# 自动格式化
pip install black
black model/ converter/ tests/
```

### JavaScript 代码

- 缩进：2 个空格
- 使用 `'use strict'`
- 函数和变量命名：camelCase
- 常量命名：UPPER_SNAKE_CASE
- 添加必要的注释，解释复杂逻辑

### HTML/CSS 代码

- 缩进：2 个空格
- 语义化 HTML 标签
- CSS 类名：kebab-case
- 响应式设计：使用 `clamp()`、`aspect-ratio`、媒体查询

---

## 提交规范

### Commit Message 格式

```
<type>: [Task X] 简要描述

详细说明（可选，每行不超过 72 字符）
```

**Type 类型**：

- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `refactor`: 代码重构（不影响功能）
- `perf`: 性能优化
- `test`: 测试相关
- `chore`: 构建/工具/依赖等杂项

**示例**：

```
feat: [Task 4] 性能优化(离屏Canvas+笔画去重+图片类型识别)

渲染优化:
- engine.js: 新增renderToOffscreen()离屏预渲染API
- 一次性同步渲染全部指令到离屏canvas

数据转换优化:
- transform.py: 新增deduplicate_strokes()笔画去重+链式合并
```

### 提交频率

- 每个 Task 一次提交
- 大功能拆分为多个小提交，每个提交应能独立运行
- 避免一个提交包含多个不相关的改动

---

## 测试要求

### 测试覆盖

所有新功能必须添加相应测试：

- **单元测试**：`tests/test_transform.py`，测试 `converter/transform.py` 的核心函数
- **集成测试**：`tests/test_integration.py`，测试完整管线
- **前端测试**：使用 Playwright 测试前端交互（可选）

### 运行测试

```bash
# 运行所有测试
python -m unittest discover tests

# 运行特定测试文件
python -m unittest tests.test_transform

# 运行特定测试类
python -m unittest tests.test_transform.TestToHex

# 详细输出
python -m unittest discover tests -v
```

### 测试编写规范

```python
class TestYourFunction(unittest.TestCase):
    """Tests for your_function()."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data = ...

    def test_basic_case(self):
        """Basic case should work correctly."""
        result = your_function(self.test_data)
        self.assertEqual(result, expected)

    def test_edge_case(self):
        """Edge case should be handled gracefully."""
        with self.assertRaises(ValueError):
            your_function(invalid_input)
```

**要求**：

- 每个测试方法都有文档字符串
- 测试名称以 `test_` 开头
- 一个测试方法只测试一个行为
- 使用 `setUp`/`tearDown` 共享测试夹具

---

## Pull Request 流程

### 1. 准备

```bash
# 同步上游
git fetch upstream
git checkout main
git merge upstream/main

# 创建分支
git checkout -b feat/your-feature
```

### 2. 开发

- 编写代码，遵循代码风格
- 添加测试
- 运行测试确保通过：`python -m unittest discover tests`

### 3. 提交

```bash
git add .
git commit -m "feat: [Task X] 简要描述"
```

### 4. 推送并创建 PR

```bash
git push origin feat/your-feature
```

在 GitHub 上创建 Pull Request，描述：
- 改了什么
- 为什么改
- 如何测试
- 截图（如果是前端改动）

### 5. Review

- 等待维护者 review
- 根据 feedback 修改
- 不要 force push，保持 commit 历史清晰

---

## 项目结构

```
AI-Stroke-Painter/
├── model/                      # 模型推理端
│   ├── inference.py            # 推理入口（RL + Lite 模式）
│   ├── network.py              # Learning-to-Paint 网络架构
│   └── download_weights.sh     # 预训练权重下载脚本
├── converter/                  # 数据转换层
│   └── transform.py            # 原始数据 → 引擎 JSON
├── renderer/                   # 渲染引擎端
│   ├── engine.js               # Canvas 2D 笔刷引擎
│   └── index.html              # 前端预览界面
├── tests/                      # 测试
│   ├── test_transform.py       # 单元测试
│   └── test_integration.py     # 集成测试
├── assets/                     # 示例素材
├── docs/                       # 文档
│   └── deployment.md           # 部署指南
├── Dockerfile                  # Docker 构建文件
├── CONTRIBUTING.md             # 本文件
├── README.md                   # 项目说明
└── .gitignore
```

### 修改约束

**绝对不能修改的契约**：

1. `converter/transform.py` 输出的 JSON 格式：
   ```json
   ["line", brushId, x, y, p...]
   ```
2. `renderer/engine.js` 的核心渲染逻辑（`applyInstruction`、`renderPainting`）

所有优化和新功能必须是**可选的**，不能破坏现有 API。

---

## 联系方式

- 提交 [Issue](https://github.com/huangkunh/AI-Stroke-Painter/issues) 报告 bug 或建议
- 提交 Pull Request 贡献代码
- 讨论：在 Issue 中使用 `discussion` 标签

感谢你的贡献！
