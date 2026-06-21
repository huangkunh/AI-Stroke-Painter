#!/usr/bin/env python3
"""
Differentiable neural renderer for stroke-based painting.

This module implements a PyTorch-based differentiable renderer that can
simulate brush strokes on a canvas while preserving gradients for
backpropagation. It is the core component that enables end-to-end training
of the painting agent via reinforcement learning (DDPG).

Architecture overview
---------------------
The renderer takes a canvas tensor (B, 3, H, W) and a batch of stroke
actions (B, A) where A >= 15, and produces a new canvas with the strokes
"painted" on it. The rendering is fully differentiable: gradients flow
through the stroke parameters (position, color, width, pressure, etc.)
back to the policy network.

Stroke parameterization (15+ dimensions)
----------------------------------------
  0-1:   stroke center (x, y) in [0, 1]
  2-3:   stroke end (x, y) in [0, 1]
  4-6:   color (r, g, b) in [0, 1]
  7:     alpha (opacity) in [0, 1]
  8:     brush radius in [0, 1] (scaled to max_radius)
  9:     pressure at start in [0, 1]
  10:    pressure at end in [0, 1]
  11-12: control point for quadratic Bezier (x, y) in [0, 1]
  13:    stroke length factor in [0, 1]
  14:    brush softness in [0, 1] (0 = hard edge, 1 = soft falloff)
  15+:   optional material/texture params (reserved)

The renderer uses a Gaussian splat approach: each stroke is sampled into
N points along a quadratic Bezier curve, and each point deposits a
Gaussian "dab" of color onto the canvas. The dabs are composited with
alpha blending, which is differentiable.

Compatibility
-------------
The output of this renderer is NOT directly compatible with the Canvas 2D
engine (engine.js). Instead, it is used during TRAINING to simulate the
painting process so the agent can learn. At inference time, the trained
agent's actions are converted to the engine JSON format via
converter/transform.py, preserving the existing contract.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default stroke parameter dimension. The renderer accepts actions of at
# least this size; extra dimensions are ignored.
DEFAULT_ACTION_DIM = 15

# Number of sample points along a stroke curve for Gaussian splatting.
DEFAULT_NUM_SAMPLES = 16

# Default canvas size for the neural renderer (smaller than the display
# canvas for training speed; the agent learns at this resolution).
DEFAULT_CANVAS_SIZE = 64


# ---------------------------------------------------------------------------
# Differentiable stroke renderer
# ---------------------------------------------------------------------------

class DifferentiableRenderer(nn.Module):
    """A differentiable brush-stroke renderer using Gaussian splatting.

    The renderer composites strokes onto a canvas by:
      1. Sampling points along a quadratic Bezier curve defined by the
         stroke's start, control, and end points.
      2. For each point, depositing a 2D Gaussian "dab" of color with
         radius proportional to the brush radius and pressure.
      3. Alpha-blending the dabs onto the canvas.

    All operations use standard PyTorch tensor ops, so gradients flow
    through the stroke parameters to the policy network.
    """

    def __init__(
        self,
        canvas_size: int = DEFAULT_CANVAS_SIZE,
        action_dim: int = DEFAULT_ACTION_DIM,
        num_samples: int = DEFAULT_NUM_SAMPLES,
        max_radius: float = 8.0,
    ):
        super().__init__()
        self.canvas_size = canvas_size
        self.action_dim = action_dim
        self.num_samples = num_samples
        self.max_radius = max_radius

        # Pre-compute the sampling parameter t in [0, 1] for the Bezier curve
        t = torch.linspace(0.0, 1.0, num_samples)
        self.register_buffer("t", t)  # (num_samples,)

        # Pre-compute Bezier basis functions for quadratic curves:
        # B(t) = (1-t)^2 * P0 + 2*(1-t)*t * P1 + t^2 * P2
        self.register_buffer("b0", (1 - t) ** 2)        # (N,)
        self.register_buffer("b1", 2 * (1 - t) * t)     # (N,)
        self.register_buffer("b2", t ** 2)              # (N,)

        # Pre-compute a coordinate grid for Gaussian splatting
        # We create a grid of size (canvas_size, canvas_size) for each dab
        coords = torch.arange(canvas_size, dtype=torch.float32)
        gy, gx = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer("grid_x", gx)  # (H, W)
        self.register_buffer("grid_y", gy)  # (H, W)

    def forward(
        self,
        canvas: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Render a batch of strokes onto the canvas.

        Parameters
        ----------
        canvas : Tensor of shape (B, 3, H, W) in [0, 1]
            The current canvas state.
        action : Tensor of shape (B, A) where A >= 15
            Stroke parameters. See module docstring for the layout.

        Returns
        -------
        Tensor of shape (B, 3, H, W) in [0, 1]
            The new canvas with the stroke painted on it.
        """
        B = canvas.shape[0]
        H = W = self.canvas_size
        device = canvas.device

        # Clamp action to [-1, 1] then map to [0, 1] for most params
        a = torch.clamp(action[:, :self.action_dim], -1.0, 1.0)
        a01 = (a + 1.0) * 0.5  # (B, A) in [0, 1]

        # --- Extract stroke parameters ---
        x0 = a01[:, 0]      # start x
        y0 = a01[:, 1]      # start y
        x2 = a01[:, 2]      # end x
        y2 = a01[:, 3]      # end y
        col_r = a01[:, 4]   # color r
        col_g = a01[:, 5]   # color g
        col_b = a01[:, 6]   # color b
        alpha = a01[:, 7]   # opacity
        radius = a01[:, 8] * self.max_radius  # brush radius in pixels
        p_start = a01[:, 9]   # pressure at start
        p_end = a01[:, 10]    # pressure at end
        cx = a01[:, 11]     # control point x (for Bezier)
        cy = a01[:, 12]     # control point y
        # stroke_len = a01[:, 13]  # reserved (affects sampling density)
        softness = a01[:, 14]  # edge softness

        # --- Sample points along the quadratic Bezier curve ---
        # P(t) = b0*P0 + b1*P1 + b2*P2
        # Shape: (B, N) for x and y
        px = (self.b0.unsqueeze(0) * (x0 * W).unsqueeze(1) +
              self.b1.unsqueeze(0) * (cx * W).unsqueeze(1) +
              self.b2.unsqueeze(0) * (x2 * W).unsqueeze(1))
        py = (self.b0.unsqueeze(0) * (y0 * H).unsqueeze(1) +
              self.b1.unsqueeze(0) * (cy * H).unsqueeze(1) +
              self.b2.unsqueeze(0) * (y2 * H).unsqueeze(1))

        # --- Compute pressure at each sample point ---
        # Linear interpolation from p_start to p_end
        p = p_start.unsqueeze(1) * self.b0.unsqueeze(0) * 1.0 + \
            p_end.unsqueeze(1) * self.b2.unsqueeze(0) * 1.0
        # Normalize (b0+b1+b2 = 1, but we want linear interp, so use t directly)
        p = p_start.unsqueeze(1) * (1 - self.t).unsqueeze(0) + \
            p_end.unsqueeze(1) * self.t.unsqueeze(0)  # (B, N)

        # --- Splat Gaussian dabs onto the canvas ---
        # For each batch element, for each sample point, create a Gaussian
        # dab and composite it onto the canvas.
        #
        # To keep this efficient, we process all samples for a batch element
        # in parallel by creating a (B, N, H, W) accumulation buffer.
        new_canvas = canvas.clone()

        for b in range(B):
            # For this batch element, splat all N dabs
            # dab_positions: (N, 2) — (x, y) in pixel coords
            dab_x = px[b]  # (N,)
            dab_y = py[b]  # (N,)
            dab_r = radius[b] * p[b]  # (N,) effective radius with pressure
            dab_color = torch.stack([col_r[b], col_g[b], col_b[b]])  # (3,)
            dab_alpha = alpha[b] * p[b]  # (N,) effective alpha with pressure
            dab_soft = softness[b]

            # Create the stroke layer by accumulating dabs
            stroke_layer = torch.zeros(3, H, W, device=device)
            weight_layer = torch.zeros(1, H, W, device=device)

            for n in range(self.num_samples):
                r = dab_r[n].clamp(min=0.5)
                sigma = r * (0.5 + dab_soft * 1.5)  # softness controls spread
                # Gaussian: exp(-((x-cx)^2 + (y-cy)^2) / (2*sigma^2))
                dx = self.grid_x - dab_x[n]
                dy = self.grid_y - dab_y[n]
                dist2 = dx * dx + dy * dy
                gaussian = torch.exp(-dist2 / (2.0 * sigma * sigma + 1e-8))
                # Scale by dab alpha
                weight = gaussian * dab_alpha[n]
                # Accumulate color (weighted average)
                stroke_layer += weight.unsqueeze(0) * dab_color.unsqueeze(-1).unsqueeze(-1)
                weight_layer += weight.unsqueeze(0)

            # Composite stroke onto canvas using alpha blending
            # new = stroke * weight + canvas * (1 - weight)
            w = torch.clamp(weight_layer, 0.0, 1.0)  # (1, H, W)
            # Avoid division by zero
            safe_w = w + 1e-8
            stroke_color = stroke_layer / safe_w  # (3, H, W) average color
            # Alpha blend
            new_canvas[b] = stroke_color * w + canvas[b] * (1 - w)

        return new_canvas

    def render_sequence(
        self,
        canvas: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, list]:
        """Render a sequence of strokes, returning the final canvas and
        intermediate canvases for visualization.

        Parameters
        ----------
        canvas : Tensor (B, 3, H, W)
            Initial canvas.
        actions : Tensor (B, T, A)
            Sequence of T stroke actions.

        Returns
        -------
        final_canvas : Tensor (B, 3, H, W)
        intermediates : list of Tensor, length T
            Canvas after each stroke.
        """
        T = actions.shape[1]
        intermediates = []
        current = canvas
        for t in range(T):
            current = self.forward(current, actions[:, t])
            intermediates.append(current)
        return current, intermediates


# ---------------------------------------------------------------------------
# Stroke parameter decoder
# ---------------------------------------------------------------------------

def decode_stroke_params(action: torch.Tensor, canvas_size: int = DEFAULT_CANVAS_SIZE,
                         max_radius: float = 8.0) -> dict:
    """Decode a raw action tensor into human-readable stroke parameters.

    This is the inverse of the encoding used by the renderer. It maps the
    [-1, 1] action space back to natural ranges, compatible with the
    converter/transform.py schema.
    """
    a = action.detach().cpu()
    if a.dim() == 1:
        a = a.unsqueeze(0)
    a01 = (torch.clamp(a[:, :15], -1, 1) + 1) * 0.5
    return {
        "x_start":      float(a01[0, 0]),
        "y_start":      float(a01[0, 1]),
        "x_end":        float(a01[0, 2]),
        "y_end":        float(a01[0, 3]),
        "color_r":      float(a01[0, 4]),
        "color_g":      float(a01[0, 5]),
        "color_b":      float(a01[0, 6]),
        "color_a":      float(a01[0, 7]),
        "brush_radius": float(a01[0, 8]) * max_radius,
        "pressure_start": float(a01[0, 9]),
        "pressure_end":   float(a01[0, 10]),
        "control_x":    float(a01[0, 11]),
        "control_y":    float(a01[0, 12]),
        "stroke_len":   float(a01[0, 13]),
        "softness":     float(a01[0, 14]),
    }


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_differentiable_renderer(
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    action_dim: int = DEFAULT_ACTION_DIM,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    max_radius: float = 8.0,
) -> DifferentiableRenderer:
    """Create a DifferentiableRenderer instance."""
    return DifferentiableRenderer(
        canvas_size=canvas_size,
        action_dim=action_dim,
        num_samples=num_samples,
        max_radius=max_radius,
    )
