"""Image-conditioned planning shared by the planning and simulation examples."""

import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

from .io import image_tensor, load_prior, load_world
from .planning import CEMConfig, MixedCEM, latent_cost, latent_rollout


def read_task(path, episode=0, start=0, goal_offset=25, simulation=False):
    """Select two frames in one episode and optionally their simulator states."""
    with h5py.File(path, "r") as data:
        if not 0 <= episode < len(data["ep_len"]):
            raise ValueError("episode index is outside the dataset")
        if (
            start < 0
            or goal_offset < 1
            or start + goal_offset >= int(data["ep_len"][episode])
        ):
            raise ValueError("current and goal frames must lie within the same episode")
        row = int(data["ep_offset"][episode]) + start
        result = {
            "current_image": data["pixels"][row],
            "goal_image": data["pixels"][row + goal_offset],
        }
        if simulation:
            if "qpos" in data and "qvel" in data:
                result.update(
                    initial_state=np.r_[data["qpos"][row], data["qvel"][row]],
                    goal_position=data["privileged_block_0_pos"][row + goal_offset],
                    goal_quaternion=data["privileged_block_0_quat"][row + goal_offset],
                )
            else:
                result.update(
                    initial_state=data["state"][row],
                    goal_state=data["state"][row + goal_offset],
                )
    return result


class ImagePlanner:
    """Keep frozen models loaded while updating the image condition each round."""

    def __init__(
        self,
        world,
        domain,
        prior=None,
        mode="mixed",
        device="cuda",
        candidates=100,
        elites=10,
        iterations=10,
        ratio=0.2,
        seed=42,
        cost_protocol="terminal",
    ):
        if domain not in {"pusht", "cube"} or mode not in {"cem", "mixed", "direct"}:
            raise ValueError("Unsupported domain or planning mode")
        if mode != "cem" and prior is None:
            raise ValueError(
                "A trained --prior is required for mixed or direct planning"
            )
        self.domain, self.mode, self.device = domain, mode, device
        self.cost_protocol = cost_protocol
        self.model = load_world(world, device)
        self.prior = load_prior(prior, device, world) if mode != "cem" else None
        scaler_path = Path(world).with_name("action_scaler.json")
        if not scaler_path.is_file():
            raise FileNotFoundError(
                "Keep action_scaler.json beside the world-model weights"
            )
        scaler = json.loads(scaler_path.read_text())
        self.model_mean = torch.tensor(
            scaler["mean"], dtype=torch.float32, device=device
        )
        self.model_scale = torch.tensor(
            scaler["scale"], dtype=torch.float32, device=device
        )
        self.dim = len(self.model_mean)
        if self.dim != (2 if domain == "pusht" else 5):
            raise ValueError("The action scaler does not match the selected domain")
        if self.prior is not None:
            default_mean = [-0.0087, 0.0068] if domain == "pusht" else scaler["mean"]
            default_scale = [0.2019, 0.2002] if domain == "pusht" else scaler["scale"]
            self.mean = torch.tensor(
                self.prior.metadata.get("action_mean", default_mean), device=device
            )
            self.scale = torch.tensor(
                self.prior.metadata.get("action_scale", default_scale), device=device
            )
        else:
            self.mean, self.scale = self.model_mean, self.model_scale
        cfg = CEMConfig(
            action_dim=self.dim * 5,
            candidates=candidates,
            elites=elites,
            iterations=iterations,
            ratio=ratio if mode == "mixed" else 0.0,
        )
        self.solver = MixedCEM(cfg, device, seed)
        self.generator = torch.Generator(device=device).manual_seed(seed)

    @torch.inference_mode()
    def plan(self, current_image, goal_image, round_index=0):
        current = self.model.encode(
            {"pixels": image_tensor(current_image, self.device)}
        )["emb"]
        goal = self.model.encode({"pixels": image_tensor(goal_image, self.device)})[
            "emb"
        ]
        cfg = self.solver.config
        count = 1 if self.mode == "direct" else cfg.prior_count * cfg.iterations
        pool = None
        if count:
            pool = self.prior.sample(
                current[:, -1].expand(count, -1),
                goal[:, -1].expand(count, -1),
                generator=self.generator,
                steps=100 if self.domain == "pusht" else 10,
            )

        def score(candidates):
            raw = candidates.reshape(-1, 5, 5, self.dim) * self.scale + self.mean
            normalized = ((raw - self.model_mean) / self.model_scale).flatten(-2)
            predictions = latent_rollout(self.model, current, normalized)
            return latent_cost(
                predictions, goal, protocol=self.cost_protocol, replan=round_index
            )

        plan, costs = (
            (pool[0], []) if self.mode == "direct" else self.solver.solve(score, pool)
        )
        actions = (plan.reshape(25, self.dim) * self.scale + self.mean).cpu().numpy()
        if self.domain == "cube":
            actions = np.clip(actions, -1.0, 1.0)
        return actions, costs


def add_planning_arguments(parser):
    parser.add_argument("--domain", choices=["pusht", "cube"], required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--mode", choices=["cem", "mixed", "direct"], default="mixed")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--elites", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cost",
        choices=["terminal", "pusht"],
        default="terminal",
        help="terminal latent cost, or the PushT staged weighting schedule",
    )


def planner_from_args(args):
    return ImagePlanner(
        args.world,
        args.domain,
        args.prior,
        args.mode,
        args.device,
        args.candidates,
        args.elites,
        args.iterations,
        args.ratio,
        args.seed,
        args.cost,
    )
