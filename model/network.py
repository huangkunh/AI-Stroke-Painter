"""
Learning-to-Paint model architecture (PyTorch).

This is a faithful, lightweight re-implementation of the network used in
"Learning to Paint With Model-based Deep Reinforcement Learning" (ICCV 2019,
Zheng et al., https://github.com/hzwer/ICCV2019-LearningToPaint).

It contains three components:
  * Renderer  : turns a stroke-action sequence into a canvas image
  * Discriminator (optional, only used during training)
  * Agent     : the policy network that emits the next stroke action

Only the Agent + Renderer are needed for inference. The Agent is a
convolutional recurrent policy that observes (target_image, current_canvas)
and outputs an action vector of dimension `action_dim = 13`:

    [x_start, y_start, x_end, y_end,
     color_r, color_g, color_b, color_a,
     brush_radius, pressure, bend_mid_x, bend_mid_y, stroke_len]

The first 9 dimensions are the ones consumed by `converter/transform.py`.
The remaining 4 are auxiliary (pressure hint, control point for curvature,
and an expected stroke length) and are kept for forward compatibility with
more advanced brush engines.

The pretrained weights released by the original authors can be downloaded
with `download_weights.sh`. When torch is unavailable or the weights are
missing, `inference.py` falls back to `LitePainter` (see inference.py) so
the full image -> strokes -> render pipeline still runs end-to-end.
"""
from __future__ import annotations

import math
from typing import Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch is optional for the lite path
    _HAS_TORCH = False


# Canvas / action constants --------------------------------------------------
CANVAS_SIZE = 512          # internal resolution the network operates on
ACTION_DIM = 13            # see module docstring
HIDDEN_DIM = 512
BRUSH_RADII = [1, 2, 3, 5, 8, 12, 20]   # discrete brush sizes the agent picks among


# ---------------------------------------------------------------------------
# Renderer (neural, differentiable) -----------------------------------------
# ---------------------------------------------------------------------------
if _HAS_TORCH:

    class _Renderer(nn.Module):
        """Neural renderer: maps (canvas, action) -> new canvas.

        A simplified version of the original renderer. It is only used to
        *simulate* the canvas during RL inference so the Agent can observe the
        effect of its own strokes. The actual visual replay is done by the
        Canvas 2D engine in /renderer, NOT by this network.
        """

        def __init__(self, action_dim: int = ACTION_DIM):
            super().__init__()
            self.action_dim = action_dim
            self.conv = nn.Sequential(
                nn.Conv2d(3 + action_dim, 32, 3, 1, 1), nn.InstanceNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 32, 3, 1, 1), nn.InstanceNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 32, 3, 1, 1), nn.InstanceNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 32, 3, 1, 1), nn.InstanceNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 32, 3, 1, 1), nn.InstanceNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 32, 3, 1, 1), nn.InstanceNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 32, 3, 1, 1), nn.InstanceNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 32, 3, 1, 1), nn.InstanceNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 32, 3, 1, 1), nn.InstanceNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 32, 3, 1, 1), nn.InstanceNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 3, 3, 1, 1),
            )

        def forward(self, canvas: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            b, _, h, w = canvas.shape
            action_map = action.view(b, self.action_dim, 1, 1).expand(b, self.action_dim, h, w)
            x = torch.cat([canvas, action_map], dim=1)
            out = self.conv(x)
            return out + canvas  # residual: paint on top of the existing canvas


# ---------------------------------------------------------------------------
# Agent (recurrent policy) --------------------------------------------------
# ---------------------------------------------------------------------------
if _HAS_TORCH:

    class _Agent(nn.Module):
        """Convolutional recurrent policy.

        Observes a 6-channel input (target RGB + canvas RGB) at each step and
        emits an action vector. A GRU keeps a hidden state across the stroke
        sequence so the agent remembers what it has already painted.
        """

        def __init__(self, action_dim: int = ACTION_DIM, hidden_dim: int = HIDDEN_DIM):
            super().__init__()
            self.action_dim = action_dim
            self.hidden_dim = hidden_dim
            self.conv = nn.Sequential(
                nn.Conv2d(6, 32, 4, 4, 0), nn.ReLU(),
                nn.Conv2d(32, 32, 4, 4, 0), nn.ReLU(),
                nn.Conv2d(32, 32, 4, 4, 0), nn.ReLU(),
                nn.Conv2d(32, 32, 4, 4, 0), nn.ReLU(),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, 6, CANVAS_SIZE, CANVAS_SIZE)
                feat_dim = self.conv(dummy).flatten(1).shape[1]
            self.fc_in = nn.Linear(feat_dim, hidden_dim)
            self.gru = nn.GRUCell(hidden_dim, hidden_dim)
            self.fc_out = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )
            # Output heads are activated by tanh (all actions are in [-1, 1])
            self.tanh = nn.Tanh()

        def forward(self, target: torch.Tensor, canvas: torch.Tensor, h: torch.Tensor):
            x = torch.cat([target, canvas], dim=1)
            feat = self.conv(x).flatten(1)
            feat = F.relu(self.fc_in(feat))
            h = self.gru(feat, h)
            raw = self.fc_out(h)
            action = self.tanh(raw)
            return action, h

        def init_hidden(self, batch: int, device) -> torch.Tensor:
            return torch.zeros(batch, self.hidden_dim, device=device)


# ---------------------------------------------------------------------------
# Public factory ------------------------------------------------------------
# ---------------------------------------------------------------------------

def build_agent(action_dim: int = ACTION_DIM, hidden_dim: int = HIDDEN_DIM):
    """Return an Agent module. Raises RuntimeError if torch is missing."""
    if not _HAS_TORCH:
        raise RuntimeError(
            "PyTorch is not installed. Install it with `pip install torch` "
            "or run inference.py in --mode lite to use the heuristic painter."
        )
    return _Agent(action_dim=action_dim, hidden_dim=hidden_dim)


def build_renderer(action_dim: int = ACTION_DIM):
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is not installed; cannot build the neural renderer.")
    return _Renderer(action_dim=action_dim)


def decode_action(action) -> dict:
    """Convert a raw action tensor/array (shape [13]) into a readable dict.

    Coordinates and colours are mapped from [-1, 1] back to their natural
    ranges. This is the canonical schema consumed by `converter/transform.py`.
    """
    a = action.detach().cpu().numpy() if _HAS_TORCH and hasattr(action, "detach") else action
    return {
        "x_start":      float((a[0]  + 1) * 0.5),   # 0..1
        "y_start":      float((a[1]  + 1) * 0.5),
        "x_end":        float((a[2]  + 1) * 0.5),
        "y_end":        float((a[3]  + 1) * 0.5),
        "color_r":      float((a[4]  + 1) * 0.5),   # 0..1
        "color_g":      float((a[5]  + 1) * 0.5),
        "color_b":      float((a[6]  + 1) * 0.5),
        "color_a":      float((a[7]  + 1) * 0.5),
        "brush_radius": float((a[8]  + 1) * 0.5) * 20.0,  # 0..20 px
        "pressure":     float((a[9]  + 1) * 0.5),   # 0..1 hint
        "bend_mid_x":   float((a[10] + 1) * 0.5),
        "bend_mid_y":   float((a[11] + 1) * 0.5),
        "stroke_len":   float((a[12] + 1) * 0.5),
    }
