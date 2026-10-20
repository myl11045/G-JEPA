"""Small simulator adapters and a receding-horizon control loop."""

import os
import sys

import numpy as np


class PushTSimulation:
    def __init__(self, task, seed):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        from .envs.pusht_wrapper import PushTWrapper

        self.env = PushTWrapper(with_velocity=True, with_target=True)
        self.goal_state = np.asarray(task["goal_state"])
        obs, self.state = self.env.prepare(
            seed, np.asarray(task["initial_state"]).copy()
        )
        self.image = obs["visual"]

    def step(self, action):
        obs, _, _, info = self.env.step(action)
        self.image, self.state = obs["visual"], info["state"]

    def metrics(self):
        result = self.env.eval_state(self.goal_state, self.state)
        return (
            float(result["block_pos_dist"]),
            float(result["block_angle_dist_deg"]),
            bool(result["success"]),
        )

    def close(self):
        self.env.close()


class CubeSimulation:
    def __init__(self, task, seed):
        os.environ.setdefault(
            "MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl"
        )
        from stable_worldmodel.envs.ogbench.cube_env import CubeEnv

        self.env = CubeEnv(
            env_type="single",
            ob_type="states",
            permute_blocks=False,
            mode="data_collection",
            terminate_at_goal=False,
            success_timing="post",
            visualize_info=False,
            pixel_transparent_arm=False,
            height=224,
            width=224,
        )
        self.goal_state = np.asarray(task["goal_position"])
        self.env.reset(
            seed=seed, options={"state": task["initial_state"], "variation": []}
        )
        self.env.set_target_pos(0, self.goal_state, task["goal_quaternion"])
        self._observe()

    def _observe(self):
        self.image = self.env.render()
        self.state = np.asarray(self.env.compute_ob_info()["privileged/block_0_pos"])

    def step(self, action):
        self.env.step(np.clip(action, -1, 1))
        self._observe()

    def metrics(self):
        distance = float(np.linalg.norm(self.state - self.goal_state))
        return distance * 100, float("nan"), distance <= 0.04

    def close(self):
        self.env.close()


def run_rollout(
    env, planner, goal_image, max_steps=100, execute=5, stop_on_success=False
):
    """Recondition the planner on actual observations after each action prefix."""
    if max_steps < 1 or not 1 <= execute <= 25:
        raise ValueError("max_steps must be positive and execute must lie in [1, 25]")
    frames, states, distances, angles, successes, actions, replans = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    def record():
        frames.append(np.asarray(env.image).copy())
        states.append(np.asarray(env.state).copy())
        distance, angle, success = env.metrics()
        distances.append(distance)
        angles.append(angle)
        successes.append(success)

    record()
    while len(actions) < max_steps and not (stop_on_success and successes[-1]):
        plan, costs = planner.plan(env.image, goal_image, round_index=len(replans))
        replans.append({"step": len(actions), "elite_cost_per_iteration": costs})
        for action in plan[: min(execute, max_steps - len(actions))]:
            env.step(action)
            actions.append(np.asarray(action).copy())
            record()
            if stop_on_success and successes[-1]:
                break
    arrays = {
        "frames": np.stack(frames),
        "goal_image": np.asarray(goal_image),
        "states": np.stack(states),
        "goal_state": np.asarray(env.goal_state),
        "position_error": np.asarray(distances),
        "angle_error_deg": np.asarray(angles),
        "success": np.asarray(successes, dtype=bool),
        "actions": np.asarray(actions),
        "steps": np.arange(len(frames)),
    }
    summary = {
        "environment_steps": len(actions),
        "reached_goal": bool(any(successes)),
        "final_success": bool(successes[-1]),
        "final_position_error": distances[-1],
        "final_angle_error_deg": angles[-1] if np.isfinite(angles[-1]) else None,
        "replans": replans,
    }
    return arrays, summary
