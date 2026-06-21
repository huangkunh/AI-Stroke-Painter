#!/usr/bin/env python3
"""
DDPG (Deep Deterministic Policy Gradient) agent for stroke-based painting.

This module implements the DDPG algorithm for training a painting agent:
  - Actor network: maps state (target + canvas) -> continuous action (stroke params)
  - Critic network: maps (state, action) -> Q-value
  - Experience replay buffer
  - Soft target updates
  - Ornstein-Uhlenbeck noise for exploration

The agent is designed to work with the differentiable renderer
(model/differentiable_renderer.py) and produces 15-dimensional stroke
actions compatible with the existing converter/transform.py pipeline.

References
----------
- Lillicrap et al., "Continuous control with deep reinforcement learning", ICLR 2016
- Zheng et al., "Learning to Paint With Model-based Deep Reinforcement Learning", ICCV 2019
"""
from __future__ import annotations

import math
import os
import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None
    nn = None
    F = None
    Adam = None
    np = None


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

DEFAULT_ACTION_DIM = 15
DEFAULT_CANVAS_SIZE = 64        # small for fast training; scale up for fine results
DEFAULT_HIDDEN_DIM = 256
DEFAULT_GAMMA = 0.95            # discount factor
DEFAULT_TAU = 0.005             # soft target update rate
DEFAULT_ACTOR_LR = 1e-4
DEFAULT_CRITIC_LR = 1e-3
DEFAULT_BUFFER_SIZE = 10000
DEFAULT_BATCH_SIZE = 16
DEFAULT_NOISE_THETA = 0.15
DEFAULT_NOISE_SIGMA = 0.20


# ---------------------------------------------------------------------------
# Actor Network
# ---------------------------------------------------------------------------

class Actor(nn.Module if _HAS_TORCH else object):
    """Actor network: state -> action.

    The state is a concatenation of the target image and current canvas
    (both B x 3 x H x W). The network uses a shared CNN backbone followed
    by an LSTM for temporal reasoning, then a linear head that outputs
    the action in [-1, 1]^A.

    The recurrent connection lets the agent remember what it has painted
    so far, which is essential for coherent multi-stroke compositions.
    """

    def __init__(self, action_dim: int = DEFAULT_ACTION_DIM,
                 hidden_dim: int = DEFAULT_HIDDEN_DIM,
                 canvas_size: int = DEFAULT_CANVAS_SIZE):
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required for Actor.")
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # CNN backbone: processes (target, canvas) pair -> feature vector
        # Input: 6 channels (3 target + 3 canvas)
        self.conv = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=5, stride=2, padding=2),   # -> 32 x H/2
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # -> 64 x H/4
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> 128 x H/8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),                            # -> 128 x 4 x 4
        )
        feat_dim = 128 * 4 * 4

        # LSTM for temporal reasoning across strokes
        self.lstm = nn.LSTMCell(feat_dim, hidden_dim)

        # Action head
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[Any, Any]:
        """Initialize LSTM hidden state."""
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)
        return h, c

    def forward(self, target: Any, canvas: Any, hidden: Tuple[Any, Any]) -> Tuple[Any, Tuple[Any, Any]]:
        """Forward pass.

        Args:
            target:  B x 3 x H x W target image
            canvas:  B x 3 x H x W current canvas
            hidden:  (h, c) LSTM hidden state

        Returns:
            action:  B x A action in [-1, 1]
            hidden:  (h, c) updated hidden state
        """
        x = torch.cat([target, canvas], dim=1)   # B x 6 x H x W
        feat = self.conv(x)                       # B x (128*4*4)
        feat = feat.view(feat.size(0), -1)
        h, c = self.lstm(feat, hidden)
        x = F.relu(self.fc1(h))
        action = torch.tanh(self.fc2(x))          # B x A in [-1, 1]
        return action, (h, c)


# ---------------------------------------------------------------------------
# Critic Network
# ---------------------------------------------------------------------------

class Critic(nn.Module if _HAS_TORCH else object):
    """Critic network: (state, action) -> Q-value.

    The state features are extracted with a CNN (shared structure with
    Actor), concatenated with the action, and passed through MLP heads
    to produce a scalar Q-value.
    """

    def __init__(self, action_dim: int = DEFAULT_ACTION_DIM,
                 hidden_dim: int = DEFAULT_HIDDEN_DIM,
                 canvas_size: int = DEFAULT_CANVAS_SIZE):
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required for Critic.")
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # CNN backbone (same architecture as Actor)
        self.conv = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        feat_dim = 128 * 4 * 4

        # LSTM for temporal context
        self.lstm = nn.LSTMCell(feat_dim, hidden_dim)

        # Q-value head: (hidden_state, action) -> Q
        self.fc1 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[Any, Any]:
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)
        return h, c

    def forward(self, target: Any, canvas: Any, action: Any,
                hidden: Tuple[Any, Any]) -> Tuple[Any, Tuple[Any, Any]]:
        """Forward pass.

        Args:
            target:  B x 3 x H x W
            canvas:  B x 3 x H x W
            action:  B x A
            hidden:  (h, c)

        Returns:
            q_value: B x 1
            hidden:  (h, c)
        """
        x = torch.cat([target, canvas], dim=1)
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        h, c = self.lstm(feat, hidden)
        x = torch.cat([h, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q, (h, c)


# ---------------------------------------------------------------------------
# Experience Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Experience replay buffer for DDPG.

    Stores (state, action, reward, next_state, done, hidden) tuples and
    supports random batch sampling. The hidden state is stored to support
    the recurrent Actor/Critic networks.
    """

    def __init__(self, capacity: int = DEFAULT_BUFFER_SIZE):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, target: Any, canvas: Any, action: Any, reward: float,
             next_canvas: Any, done: bool, hidden_h: Any, hidden_c: Any):
        """Add a transition to the buffer."""
        self.buffer.append((
            target.cpu() if _HAS_TORCH and hasattr(target, 'cpu') else target,
            canvas.cpu() if _HAS_TORCH and hasattr(canvas, 'cpu') else canvas,
            action.cpu() if _HAS_TORCH and hasattr(action, 'cpu') else action,
            float(reward),
            next_canvas.cpu() if _HAS_TORCH and hasattr(next_canvas, 'cpu') else next_canvas,
            bool(done),
            hidden_h.cpu() if _HAS_TORCH and hasattr(hidden_h, 'cpu') else hidden_h,
            hidden_c.cpu() if _HAS_TORCH and hasattr(hidden_c, 'cpu') else hidden_c,
        ))

    def sample(self, batch_size: int, device: Any) -> Tuple:
        """Sample a random batch of transitions."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        targets, canvases, actions, rewards, next_canvases, dones, hh, hc = zip(*batch)
        return (
            torch.stack(targets).to(device),
            torch.stack(canvases).to(device),
            torch.stack(actions).to(device),
            torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1),
            torch.stack(next_canvases).to(device),
            torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1),
            torch.stack(hh).to(device),
            torch.stack(hc).to(device),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck Noise (exploration)
# ---------------------------------------------------------------------------

class OUNoise:
    """Ornstein-Uhlenbeck process for exploration noise.

    Produces temporally correlated noise that is well-suited for continuous
    control tasks. The noise mean-reverts to zero with rate theta and has
    volatility sigma.
    """

    def __init__(self, action_dim: int = DEFAULT_ACTION_DIM,
                 theta: float = DEFAULT_NOISE_THETA,
                 sigma: float = DEFAULT_NOISE_SIGMA,
                 mu: float = 0.0):
        self.action_dim = action_dim
        self.theta = theta
        self.sigma = sigma
        self.mu = mu
        self.state = np.ones(action_dim) * mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self) -> np.ndarray:
        """Generate a noise sample."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state = self.state + dx
        return self.state


# ---------------------------------------------------------------------------
# DDPG Agent
# ---------------------------------------------------------------------------

class DDPGAgent:
    """DDPG agent for stroke-based painting.

    Combines Actor, Critic, replay buffer, and target networks into a
    complete agent that can be trained with the differentiable renderer.

    Usage:
        agent = DDPGAgent(device='cpu')
        # Training loop:
        for episode in range(num_episodes):
            state = env.reset()
            hidden = agent.init_hidden(batch_size=1)
            for step in range(max_steps):
                action = agent.select_action(state, hidden, noise=True)
                next_state, reward, done = env.step(action)
                agent.replay_buffer.push(...)
                agent.update(batch_size=16)
    """

    def __init__(self,
                 action_dim: int = DEFAULT_ACTION_DIM,
                 hidden_dim: int = DEFAULT_HIDDEN_DIM,
                 canvas_size: int = DEFAULT_CANVAS_SIZE,
                 gamma: float = DEFAULT_GAMMA,
                 tau: float = DEFAULT_TAU,
                 actor_lr: float = DEFAULT_ACTOR_LR,
                 critic_lr: float = DEFAULT_CRITIC_LR,
                 buffer_size: int = DEFAULT_BUFFER_SIZE,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 device: str = 'cpu'):
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required for DDPGAgent.")

        self.device = torch.device(device)
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Main networks
        self.actor = Actor(action_dim, hidden_dim, canvas_size).to(self.device)
        self.critic = Critic(action_dim, hidden_dim, canvas_size).to(self.device)

        # Target networks (soft-updated copies)
        self.actor_target = Actor(action_dim, hidden_dim, canvas_size).to(self.device)
        self.critic_target = Critic(action_dim, hidden_dim, canvas_size).to(self.device)
        self._hard_update(self.actor, self.actor_target)
        self._hard_update(self.critic, self.critic_target)

        # Optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)

        # Replay buffer and noise
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.noise = OUNoise(action_dim)

        # Training stats
        self.actor_losses: List[float] = []
        self.critic_losses: List[float] = []

    def init_hidden(self, batch_size: int = 1) -> Tuple[Any, Any]:
        """Initialize LSTM hidden states for both Actor and Critic."""
        return self.actor.init_hidden(batch_size, self.device)

    def select_action(self, target: Any, canvas: Any,
                      hidden: Tuple[Any, Any],
                      noise: bool = False,
                      noise_scale: float = 1.0) -> Tuple[Any, Tuple[Any, Any]]:
        """Select an action using the current policy.

        Args:
            target:       1 x 3 x H x W target image
            canvas:       1 x 3 x H x W current canvas
            hidden:       (h, c) Actor hidden state
            noise:        if True, add OU noise for exploration
            noise_scale:  scaling factor for the noise

        Returns:
            action:  1 x A action in [-1, 1]
            hidden:  updated (h, c)
        """
        with torch.no_grad():
            action, hidden = self.actor(target, canvas, hidden)
        if noise:
            n = self.noise.sample()
            action = action + noise_scale * torch.tensor(n, dtype=action.dtype,
                                                          device=action.device)
            action = torch.clamp(action, -1.0, 1.0)
        return action, hidden

    def update(self, batch_size: Optional[int] = None) -> Tuple[float, float]:
        """Perform one DDPG update step.

        Samples a batch from the replay buffer and updates both Actor and
        Critic networks via gradient descent.

        Returns:
            (critic_loss, actor_loss) as floats, or (0, 0) if buffer too small.
        """
        bs = batch_size or self.batch_size
        if len(self.replay_buffer) < bs:
            return 0.0, 0.0

        (targets, canvases, actions, rewards, next_canvases, dones,
         hh, hc) = self.replay_buffer.sample(bs, self.device)

        # ---- Critic update ----
        # Compute target Q: r + gamma * Q'(s', mu'(s')) * (1 - done)
        with torch.no_grad():
            h_t = hh.clone()
            c_t = hc.clone()
            next_actions = []
            h_next, c_next = h_t, c_t
            # Single-step action prediction (simplified; full sequence would
            # unroll the LSTM, but for buffer samples we use the stored hidden)
            next_action, _ = self.actor_target(targets, next_canvases, (h_next, c_next))
            target_q, _ = self.critic_target(targets, next_canvases, next_action, (h_next, c_next))
            target_q = rewards + self.gamma * target_q * (1.0 - dones)

        current_q, _ = self.critic(targets, canvases, actions, (hh, hc))
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # ---- Actor update ----
        # Maximize Q(s, mu(s)) = minimize -Q(s, mu(s))
        pred_action, _ = self.actor(targets, canvases, (hh, hc))
        actor_q, _ = self.critic(targets, canvases, pred_action, (hh, hc))
        actor_loss = -actor_q.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # ---- Soft target update ----
        self._soft_update(self.actor, self.actor_target, self.tau)
        self._soft_update(self.critic, self.critic_target, self.tau)

        self.critic_losses.append(float(critic_loss.item()))
        self.actor_losses.append(float(actor_loss.item()))
        return float(critic_loss.item()), float(actor_loss.item())

    def save(self, path: str):
        """Save agent state to a file."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load agent state from a file."""
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.actor_target.load_state_dict(ckpt['actor_target'])
        self.critic_target.load_state_dict(ckpt['critic_target'])
        self.actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
        self.critic_optimizer.load_state_dict(ckpt['critic_optimizer'])

    @staticmethod
    def _hard_update(source: nn.Module, target: nn.Module):
        """Copy weights from source to target."""
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(sp.data)

    @staticmethod
    def _soft_update(source: nn.Module, target: nn.Module, tau: float):
        """Soft update: target = tau * source + (1 - tau) * target."""
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_ddpg_agent(**kwargs) -> DDPGAgent:
    """Create a DDPGAgent with default or custom hyperparameters."""
    return DDPGAgent(**kwargs)
