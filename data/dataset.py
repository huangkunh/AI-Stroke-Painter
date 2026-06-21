#!/usr/bin/env python3
"""
Image dataset and training environment for the painting agent.

This module provides:
  - PaintDataset: loads images from a directory for training
  - PaintEnv: a gym-like RL environment that wraps the differentiable renderer
  - RewardFunction: computes reward based on SSIM, LPIPS, MSE, etc.

The environment follows the standard RL interface:
  state = env.reset(target_image)
  action = agent.select_action(state)
  next_state, reward, done, info = env.step(action)
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None
    nn = None
    F = None
    Dataset = object
    DataLoader = None


# ---------------------------------------------------------------------------
# Image dataset
# ---------------------------------------------------------------------------

class PaintDataset(Dataset):
    """Dataset that loads and preprocesses images for painting training.

    Images are resized to (canvas_size, canvas_size) and normalized to [0, 1].
    Supports any image format readable by PIL (jpg, png, bmp, webp, etc.).
    """

    def __init__(self, image_dir: str, canvas_size: int = 64,
                 max_images: Optional[int] = None,
                 augment: bool = True):
        """Initialize the dataset.

        Args:
            image_dir: Directory containing training images.
            canvas_size: Target image size (square).
            max_images: Maximum number of images to load (None = all).
            augment: Whether to apply data augmentation (flip, color jitter).
        """
        self.canvas_size = canvas_size
        self.augment = augment
        self.image_paths: List[str] = []
        for root, _, files in os.walk(image_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    self.image_paths.append(os.path.join(root, f))
        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load and preprocess an image.

        Returns:
            Tensor of shape (3, canvas_size, canvas_size) in [0, 1].
        """
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = img.resize((self.canvas_size, self.canvas_size), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0

        if self.augment:
            arr = self._augment(arr)

        # HWC -> CHW
        return torch.from_numpy(arr).permute(2, 0, 1)

    def _augment(self, arr: np.ndarray) -> np.ndarray:
        """Apply random augmentation."""
        # Random horizontal flip
        if np.random.random() < 0.5:
            arr = arr[:, ::-1].copy()
        # Random vertical flip
        if np.random.random() < 0.3:
            arr = arr[::-1].copy()
        # Color jitter
        if np.random.random() < 0.5:
            brightness = np.random.uniform(0.9, 1.1)
            arr = np.clip(arr * brightness, 0, 1)
        if np.random.random() < 0.5:
            contrast = np.random.uniform(0.9, 1.1)
            mean = arr.mean()
            arr = np.clip((arr - mean) * contrast + mean, 0, 1)
        return arr


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

class RewardFunction:
    """Computes reward based on image similarity metrics.

    The reward is a weighted combination of:
      - MSE improvement (pixel-level)
      - SSIM (structural similarity)
      - LPIPS (learned perceptual similarity, if available)

    Higher reward = better painting progress.
    """

    def __init__(self, w_mse: float = 1.0, w_ssim: float = 2.0,
                 w_lpips: float = 0.5, use_lpips: bool = False):
        self.w_mse = w_mse
        self.w_ssim = w_ssim
        self.w_lpips = w_lpips
        self.use_lpips = use_lpips
        self._lpips_net = None
        if use_lpips:
            try:
                import lpips
                self._lpips_net = lpips.LPIPS(net='alex')
            except ImportError:
                print("[reward] lpips not available, falling back to MSE+SSIM")
                self.use_lpips = False

    def compute(self, canvas: torch.Tensor, target: torch.Tensor,
                prev_canvas: Optional[torch.Tensor] = None) -> Tuple[float, Dict]:
        """Compute reward for the current canvas state.

        Args:
            canvas: Current canvas (B, 3, H, W) in [0, 1].
            target: Target image (B, 3, H, W) in [0, 1].
            prev_canvas: Previous canvas for delta computation.

        Returns:
            (reward, info_dict)
        """
        info: Dict[str, float] = {}
        batch_size = canvas.shape[0]

        # MSE (lower is better)
        mse = F.mse_loss(canvas, target).item()
        info['mse'] = mse

        # SSIM (higher is better, range [-1, 1])
        ssim_val = self._batch_ssim(canvas, target).mean().item()
        info['ssim'] = ssim_val

        # LPIPS (lower is better)
        lpips_val = 0.0
        if self.use_lpips and self._lpips_net is not None:
            # LPIPS expects [-1, 1] range
            lpips_val = self._lpips_net(
                canvas * 2 - 1, target * 2 - 1
            ).mean().item()
            info['lpips'] = lpips_val

        # Reward: negative MSE + positive SSIM - LPIPS
        reward = -self.w_mse * mse + self.w_ssim * ssim_val
        if self.use_lpips:
            reward -= self.w_lpips * lpips_val

        # Delta reward (improvement over previous canvas)
        if prev_canvas is not None:
            prev_mse = F.mse_loss(prev_canvas, target).item()
            info['delta_mse'] = prev_mse - mse  # positive = improvement
            reward += 0.5 * (prev_mse - mse) * 10  # amplify improvement

        info['reward'] = reward
        return reward, info

    def _batch_ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute SSIM for a batch of images (simplified)."""
        if not _HAS_TORCH:
            return torch.tensor(0.0)
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        mu_x = F.avg_pool2d(x, kernel_size=8, stride=1, padding=3)
        mu_y = F.avg_pool2d(y, kernel_size=8, stride=1, padding=3)
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        sigma_x_sq = F.avg_pool2d(x ** 2, 8, 1, 3) - mu_x_sq
        sigma_y_sq = F.avg_pool2d(y ** 2, 8, 1, 3) - mu_y_sq
        sigma_xy = F.avg_pool2d(x * y, 8, 1, 3) - mu_xy
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        return ssim_map.mean(dim=[1, 2, 3])


# ---------------------------------------------------------------------------
# Painting environment
# ---------------------------------------------------------------------------

class PaintEnv:
    """RL environment for painting.

    State: (target_image, current_canvas) — both (3, H, W) tensors.
    Action: 15-dimensional stroke parameters in [-1, 1].
    Reward: based on image similarity improvement.
    Done: after max_strokes strokes.
    """

    def __init__(self, renderer, reward_fn: RewardFunction,
                 max_strokes: int = 40, device: str = 'cpu'):
        """Initialize the environment.

        Args:
            renderer: A DifferentiableRenderer instance.
            reward_fn: A RewardFunction instance.
            max_strokes: Maximum strokes per episode.
            device: torch device.
        """
        self.renderer = renderer
        self.reward_fn = reward_fn
        self.max_strokes = max_strokes
        self.device = torch.device(device)
        self.target: Optional[torch.Tensor] = None
        self.canvas: Optional[torch.Tensor] = None
        self.prev_canvas: Optional[torch.Tensor] = None
        self.step_count: int = 0

    def reset(self, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reset the environment with a new target image.

        Args:
            target: Target image (B, 3, H, W) or (3, H, W) in [0, 1].

        Returns:
            (target, canvas) — canvas starts as black.
        """
        if target.dim() == 3:
            target = target.unsqueeze(0)
        self.target = target.to(self.device)
        self.canvas = torch.zeros_like(self.target)
        self.prev_canvas = None
        self.step_count = 0
        return self.target, self.canvas

    def step(self, action: torch.Tensor) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor], float, bool, Dict
    ]:
        """Apply a stroke action and return the new state.

        Args:
            action: Stroke parameters (B, 15) in [-1, 1].

        Returns:
            (state, reward, done, info)
            state = (target, new_canvas)
        """
        assert self.target is not None and self.canvas is not None
        self.prev_canvas = self.canvas.clone()
        self.canvas = self.renderer(self.canvas, action)
        reward, info = self.reward_fn.compute(
            self.canvas, self.target, self.prev_canvas
        )
        self.step_count += 1
        done = self.step_count >= self.max_strokes
        info['step'] = self.step_count
        return (self.target, self.canvas), reward, done, info

    def get_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current state without stepping."""
        return self.target, self.canvas


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class Trainer:
    """Trains a DDPG agent to paint images.

    Usage:
        renderer = build_differentiable_renderer(canvas_size=64)
        agent = build_ddpg_agent(canvas_size=64)
        dataset = PaintDataset('assets/samples', canvas_size=64)
        trainer = Trainer(agent, renderer, dataset, device='cpu')
        trainer.train(num_episodes=100)
    """

    def __init__(self, agent, renderer, dataset: PaintDataset,
                 device: str = 'cpu', max_strokes: int = 40,
                 log_interval: int = 10, save_dir: str = 'checkpoints'):
        self.agent = agent
        self.renderer = renderer
        self.dataset = dataset
        self.device = torch.device(device)
        self.max_strokes = max_strokes
        self.log_interval = log_interval
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.reward_fn = RewardFunction(use_lpips=False)
        self.env = PaintEnv(renderer, self.reward_fn, max_strokes, device)
        self.episode_rewards: List[float] = []
        self.episode_losses: List[Tuple[float, float]] = []

    def train(self, num_episodes: int = 100,
              update_interval: int = 1) -> Dict[str, List]:
        """Run the training loop.

        Args:
            num_episodes: Number of episodes to train.
            update_interval: Update agent every N steps.

        Returns:
            Training history dict.
        """
        from ddpg_agent import OUNoise
        noise = OUNoise(self.agent.action_dim)

        for ep in range(num_episodes):
            # Sample a random target image
            idx = np.random.randint(len(self.dataset))
            target = self.dataset[idx].to(self.device)
            target, canvas = self.env.reset(target)

            # Init agent hidden state
            h, c = self.agent.actor.init_hidden(1, self.device)
            critic_h, critic_c = self.agent.critic.init_hidden(1, self.device)

            ep_reward = 0.0
            ep_critic_loss = 0.0
            ep_actor_loss = 0.0
            update_count = 0

            for step in range(self.max_strokes):
                # Select action
                action, (h, c) = self.agent.select_action(
                    target, canvas, (h, c), noise=True, noise_scale=1.0
                )

                # Take a step in the environment
                (next_target, next_canvas), reward, done, info = self.env.step(action)
                ep_reward += reward

                # Get Q-value for current state-action
                q, (critic_h, critic_c) = self.agent.critic(
                    target, canvas, action, (critic_h, critic_c)
                )

                # Get next Q-value (for TD target)
                with torch.no_grad():
                    next_action, _ = self.agent.actor_target(
                        next_target, next_canvas,
                        self.agent.actor_target.init_hidden(1, self.device)
                    )
                    next_q, _ = self.agent.critic_target(
                        next_target, next_canvas, next_action,
                        self.agent.critic_target.init_hidden(1, self.device)
                    )

                # Store transition in replay buffer
                self.agent.replay_buffer.push(
                    canvas.squeeze(0).cpu(),
                    target.squeeze(0).cpu(),
                    action.squeeze(0).cpu(),
                    reward,
                    next_canvas.squeeze(0).cpu(),
                    done,
                    h.squeeze(0).cpu(),
                    c.squeeze(0).cpu(),
                )

                # Update agent
                if step % update_interval == 0 and len(self.agent.replay_buffer) >= self.agent.batch_size:
                    cl, al = self.agent.update(self.agent.batch_size)
                    ep_critic_loss += cl
                    ep_actor_loss += al
                    update_count += 1

                canvas = next_canvas

            avg_cl = ep_critic_loss / max(update_count, 1)
            avg_al = ep_actor_loss / max(update_count, 1)
            self.episode_rewards.append(ep_reward)
            self.episode_losses.append((avg_cl, avg_al))

            if (ep + 1) % self.log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-self.log_interval:])
                print(f"[train] Episode {ep+1}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.4f} | "
                      f"Critic Loss: {avg_cl:.4f} | Actor Loss: {avg_al:.4f}")

            # Save checkpoint
            if (ep + 1) % 50 == 0:
                path = os.path.join(self.save_dir, f'agent_ep{ep+1}.pth')
                self.agent.save(path)
                print(f"[train] Saved checkpoint: {path}")

        return {
            'rewards': self.episode_rewards,
            'losses': self.episode_losses,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_training_env(image_dir: str, canvas_size: int = 64,
                       max_strokes: int = 40, device: str = 'cpu') -> Tuple:
    """Build a complete training environment.

    Returns:
        (renderer, agent, dataset, trainer)
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required for training")

    from differentiable_renderer import build_differentiable_renderer
    from ddpg_agent import build_ddpg_agent

    renderer = build_differentiable_renderer(canvas_size=canvas_size)
    agent = build_ddpg_agent(canvas_size=canvas_size, device=device)
    dataset = PaintDataset(image_dir, canvas_size=canvas_size)
    trainer = Trainer(agent, renderer, dataset, device=device, max_strokes=max_strokes)
    return renderer, agent, dataset, trainer
