from pathlib import Path

import pytest
import torch
from torch import nn

from gjepa.s4d import S4DDecoder
from gjepa.action_prior import DiffusionActor
from gjepa.planning import CEMConfig, MixedCEM, latent_rollout, latent_cost

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def test_temporal_causality_and_independent_branches():
    torch.manual_seed(1)
    head = S4DDecoder(192, 7).eval()
    assert sum(p.numel() for p in head.parameters()) == 55431
    observation = torch.randn(2, 3, 192)
    prediction = torch.randn(2, 3, 192)
    expected = head(prediction)
    head(observation)
    torch.testing.assert_close(head(prediction), expected, rtol=0, atol=0)
    altered = prediction.clone()
    altered[:, 2] += torch.randn_like(altered[:, 2])
    torch.testing.assert_close(head(altered)[:, :2], expected[:, :2], rtol=0, atol=0)


def test_physical_supervision_backpropagates_to_both_sequences():
    head = S4DDecoder(192, 7)
    observation = torch.randn(2, 3, 192, requires_grad=True)
    predicted = torch.randn(2, 3, 192, requires_grad=True)
    loss = head(observation).square().mean() + head(predicted).square().mean()
    loss.backward()
    for value in [observation.grad, predicted.grad]:
        assert torch.isfinite(value).all() and value.abs().sum() > 0


def test_prior_only_cem_retains_scoring_and_uses_fresh_candidates():
    config = CEMConfig(
        horizon=1, action_dim=1, candidates=4, elites=2, iterations=2, ratio=1
    )
    pool = torch.tensor([4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0]).reshape(8, 1, 1)
    seen = []

    def cost(actions):
        seen.append(actions.clone())
        return actions[:, 0, 0]

    result, _ = MixedCEM(config).solve(cost, pool)
    torch.testing.assert_close(result, torch.tensor([[5.5]]))
    torch.testing.assert_close(seen[0], pool[:4])
    torch.testing.assert_close(seen[1], pool[4:])


def test_mixed_candidates_and_shared_elites():
    config = CEMConfig(
        horizon=2, action_dim=1, candidates=10, elites=2, iterations=1, ratio=0.2
    )
    pool = torch.tensor([[-10.0, -10.0], [-8.0, -8.0]]).reshape(2, 2, 1)
    seen = []

    def cost(actions):
        seen.append(actions.clone())
        return actions.mean((1, 2))

    result, _ = MixedCEM(config, seed=10).solve(cost, pool)
    assert seen[0].shape == (10, 2, 1)
    torch.testing.assert_close(seen[0][0], torch.zeros(2, 1))
    torch.testing.assert_close(seen[0][1:3], pool)
    torch.testing.assert_close(result, torch.full((2, 1), -9.0))


def test_longer_horizon_keeps_gaussian_tail():
    config = CEMConfig(
        horizon=3, action_dim=1, candidates=4, elites=2, iterations=1, ratio=0.5
    )
    pool = torch.full((2, 1, 1), 10.0)
    seen = []
    MixedCEM(config).solve(
        lambda x: seen.append(x.clone()) or x.square().mean((1, 2)), pool
    )
    torch.testing.assert_close(seen[0][1:3, 0], pool[:, 0])
    assert not torch.all(seen[0][1:3, 1:] == 10.0)


def test_cost_schedule_and_no_physical_readout():
    pred = torch.arange(6.0).reshape(1, 6, 1)
    goal = torch.zeros(1, 1, 1)
    assert latent_cost(pred, goal, "pusht", 4) == 25
    weights = 2.0 ** torch.arange(6)
    weights /= weights.sum()
    torch.testing.assert_close(
        latent_cost(pred, goal, "pusht", 5), ((pred[:, :, 0] ** 2) * weights).mean(1)
    )

    class World:
        action_encoder = nn.Identity()

        def predict(self, z, action):
            return z + action

        def state_head(self, x):
            raise AssertionError("State decoder must not be called during planning")

    actions = torch.ones(2, 3, 1)
    out = latent_rollout(World(), torch.zeros(1, 1, 1), actions)
    torch.testing.assert_close(out[0, :, 0], torch.tensor([0.0, 1.0, 2.0, 3.0]))


@pytest.mark.parametrize("action_dim", [2, 5])
def test_prior_sampling_is_finite_bounded_and_seeded(action_dim):
    model = DiffusionActor(width=32, blocks=1, action_dim=action_dim).eval()
    z = torch.randn(3, 192)
    g = torch.randn(3, 192)
    first = model.sample(z, g, generator=torch.Generator().manual_seed(42), steps=10)
    second = model.sample(z, g, generator=torch.Generator().manual_seed(42), steps=10)
    assert first.shape == (3, 5, action_dim * 5)
    assert torch.isfinite(first).all() and first.abs().max() <= 5.00001
    torch.testing.assert_close(first, second, rtol=0, atol=0)
