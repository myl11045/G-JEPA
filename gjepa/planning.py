"""Mixed-proposal CEM with a shared latent objective and elite update."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CEMConfig:
    horizon: int = 5
    action_dim: int = 10
    candidates: int = 100
    elites: int = 10
    iterations: int = 10
    ratio: float = 0.2
    initial_std: float = 1.0
    sampling: str = "cpu"
    selection: str = "argsort"

    def __post_init__(self):
        if not 0 <= self.ratio <= 1:
            raise ValueError("ratio must lie in [0, 1]")
        if not 2 <= self.elites <= self.candidates:
            raise ValueError("CEM requires at least two elites and enough candidates")
        if min(self.horizon, self.action_dim, self.iterations) < 1:
            raise ValueError(
                "horizon, action dimension and iterations must be positive"
            )

    @property
    def prior_count(self):
        if self.ratio == 1:
            return self.candidates
        return min(round(self.ratio * self.candidates), self.candidates - 1)


class MixedCEM:
    def __init__(self, config, device="cpu", seed=42):
        self.config, self.device = config, torch.device(device)
        self.sample_device = (
            torch.device("cpu") if config.sampling == "cpu" else self.device
        )
        self.generator = torch.Generator(device=self.sample_device).manual_seed(seed)

    @torch.inference_mode()
    def solve(self, cost_fn, prior_pool=None, initial_mean=None):
        c = self.config
        mean = torch.zeros(c.horizon, c.action_dim, device=self.device)
        if initial_mean is not None:
            mean[: len(initial_mean)] = initial_mean.to(self.device)
        std = torch.full_like(mean, c.initial_std)
        if c.prior_count:
            if (
                prior_pool is None
                or prior_pool.shape[0] != c.prior_count * c.iterations
            ):
                raise ValueError(
                    "Supply fresh prior candidates for every CEM iteration"
                )
            if prior_pool.shape[1] > c.horizon or prior_pool.shape[2] != c.action_dim:
                raise ValueError(
                    "Prior horizon/action dimension does not match the planner"
                )
        history = []
        for iteration in range(c.iterations):
            # Draw the entire population before replacing slots, preserving the
            # Gaussian RNG stream across proposal-ratio comparisons.
            candidates = torch.randn(
                c.candidates,
                c.horizon,
                c.action_dim,
                generator=self.generator,
                device=self.sample_device,
            ).to(self.device)
            candidates = candidates * std + mean
            candidates[0] = mean
            begin = int(c.ratio < 1)
            if c.prior_count:
                pool = prior_pool[
                    iteration * c.prior_count : (iteration + 1) * c.prior_count
                ]
                candidates[begin : begin + c.prior_count, : pool.shape[1]] = pool
            costs = cost_fn(candidates)
            if costs.shape != (c.candidates,) or not torch.isfinite(costs).all():
                raise ValueError("The world model returned invalid candidate costs")
            indices = (
                torch.argsort(costs)[: c.elites]
                if c.selection == "argsort"
                else torch.topk(costs, c.elites, largest=False).indices
            )
            selected = candidates[indices]
            mean, std = selected.mean(0), selected.std(0)
            history.append(float(costs[indices].mean()))
        return mean, history


def latent_rollout(model, initial, actions, context=3):
    """Autoregressive rollout including the initial embedding, without state decoding."""
    z = initial.expand(actions.shape[0], -1, -1)
    for step in range(actions.shape[1]):
        action_embedding = model.action_encoder(actions[:, : step + 1])
        prediction = model.predict(z[:, -context:], action_embedding[:, -context:])[
            :, -1:
        ]
        z = torch.cat([z, prediction], dim=1)
    return z


def latent_cost(predictions, goal, protocol="terminal", replan=0):
    error = (predictions - goal).square().mean(-1)
    if protocol == "terminal" or replan < predictions.shape[1] - 1:
        return error[:, -1]
    if protocol != "pusht":
        raise ValueError(protocol)
    weights = 2.0 ** torch.arange(error.shape[1], device=error.device)
    weights = weights / weights.sum()
    return (error * weights).mean(-1)
