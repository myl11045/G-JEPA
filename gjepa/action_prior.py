"""Latent-conditioned diffusion action prior with deterministic DDIM sampling."""

import math
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.mod = nn.Linear(width, 2 * width)
        self.net = nn.Sequential(
            nn.Linear(width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width)
        )

    def forward(self, x, c):
        scale, shift = self.mod(c).chunk(2, -1)
        return (x + self.net(self.norm(x) * (1 + scale) + shift)) / math.sqrt(2)


class DiffusionActor(nn.Module):
    def __init__(
        self,
        width=512,
        blocks=3,
        diffusion_steps=100,
        action_dim=2,
        horizon=5,
        action_chunk=5,
    ):
        super().__init__()
        self.config = dict(
            width=width,
            blocks=blocks,
            diffusion_steps=diffusion_steps,
            action_dim=action_dim,
            horizon=horizon,
            action_chunk=action_chunk,
        )
        self.action_dim, self.horizon, self.action_chunk = (
            action_dim,
            horizon,
            action_chunk,
        )
        self.output_dim = action_dim * horizon * action_chunk
        self.register_buffer("z_mean", torch.zeros(192))
        self.register_buffer("z_std", torch.ones(192))
        self.register_buffer("action_low", torch.full((self.output_dim,), -5.0))
        self.register_buffer("action_high", torch.full((self.output_dim,), 5.0))
        t = torch.arange(diffusion_steps + 1, dtype=torch.float64) / diffusion_steps
        abar = torch.cos((t + 0.008) / 1.008 * math.pi / 2).square()
        abar = abar / abar[0]
        beta = (1 - abar[1:] / abar[:-1]).clamp(0.0001, 0.999)
        self.register_buffer("alpha_bar", torch.cumprod(1 - beta, 0).float())
        self.inp = nn.Linear(self.output_dim, width)
        self.condition = nn.Sequential(
            nn.Linear(192 * 3 + 64, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.blocks = nn.ModuleList([ResidualBlock(width) for _ in range(blocks)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, self.output_dim))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, noisy, z_current, z_goal, t):
        cur = (z_current - self.z_mean) / self.z_std
        goal = (z_goal - self.z_mean) / self.z_std
        freq = torch.exp(-math.log(10000) * torch.arange(32, device=t.device) / 32)
        phase = t.float()[:, None] * freq[None]
        context = self.condition(
            torch.cat([cur, goal, goal - cur, phase.sin(), phase.cos()], -1)
        )
        x = self.inp(noisy)
        for block in self.blocks:
            x = block(x, context)
        return self.out(x)

    def noise(self, actions, t, epsilon):
        a = self.alpha_bar[t, None]
        return a.sqrt() * actions + (1 - a).sqrt() * epsilon

    @torch.no_grad()
    def sample(self, z_current, z_goal, *, generator, steps=100, epsilon_fn=None):
        if not 2 <= steps <= len(self.alpha_bar):
            raise ValueError(steps)
        x = torch.randn(
            (len(z_current), self.output_dim),
            device=z_current.device,
            generator=generator,
        )
        grid = (
            torch.linspace(len(self.alpha_bar) - 1, 0, steps, device=x.device)
            .round()
            .long()
        )
        for i, k in enumerate(grid):
            t = k.expand(len(x))
            a = self.alpha_bar[k]
            eps = (
                self(x, z_current, z_goal, t)
                if epsilon_fn is None
                else epsilon_fn(x, t)
            )
            clean = (x - (1 - a).sqrt() * eps) / a.sqrt()
            clean = torch.maximum(
                torch.minimum(clean, self.action_high), self.action_low
            )
            # Recompute epsilon after physical-bound projection for DDIM consistency.
            eps = (x - a.sqrt() * clean) / (1 - a).sqrt()
            previous = (
                self.alpha_bar[grid[i + 1]] if i + 1 < len(grid) else x.new_tensor(1.0)
            )
            x = previous.sqrt() * clean + (1 - previous).sqrt() * eps
        return x.reshape(-1, self.horizon, self.action_dim * self.action_chunk)
