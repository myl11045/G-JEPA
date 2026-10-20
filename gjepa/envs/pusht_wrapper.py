import numpy as np
from .pusht_env import PushTEnv


def aggregate_dct(items):
    return {key: np.stack([item[key] for item in items]) for key in items[0]}


class PushTWrapper(PushTEnv):
    def __init__(
        self,
        with_velocity=True,
        with_target=True,
        color="LightSlateGray",
        agent_color="RoyalBlue",
        goal_color=None,
    ):
        super().__init__(
            with_velocity=with_velocity,
            with_target=with_target,
            color=color,
            agent_color=agent_color,
            goal_color=goal_color,
        )
        self.action_dim = self.action_space.shape[0]

    def sample_random_init_goal_states(self, seed):
        """
        Return two random states: one as the initial state and one as the goal state.
        """
        rs = np.random.RandomState(seed)

        def generate_state():
            if self.with_velocity:
                return np.array(
                    [
                        rs.randint(50, 450),
                        rs.randint(50, 450),
                        rs.randint(100, 400),
                        rs.randint(100, 400),
                        rs.randn() * 2 * np.pi - np.pi,
                        0,
                        0,  # agent velocities default 0
                    ]
                )
            else:
                return np.array(
                    [
                        rs.randint(50, 450),
                        rs.randint(50, 450),
                        rs.randint(100, 400),
                        rs.randint(100, 400),
                        rs.randn() * 2 * np.pi - np.pi,
                    ]
                )

        init_state = generate_state()
        goal_state = generate_state()

        return init_state, goal_state

    def update_env(self, env_info):
        if "shape" in env_info:
            self.shape = env_info["shape"]
        if "color" in env_info:
            self.color = env_info["color"]
        if "agent_color" in env_info:
            self.agent_color = env_info["agent_color"]
        if "goal_color" in env_info:
            self.goal_color_override = env_info["goal_color"]

    def eval_state(self, goal_state, cur_state):
        """
        Return True if the goal is reached
        [agent_x, agent_y, T_x, T_y, angle, agent_vx, agent_vy]
        """
        # if position difference is < 20, and angle difference < np.pi/9, then success
        # Block-only success (standard PushT): only block xy + angle, NOT agent.
        pos_diff = np.linalg.norm(goal_state[2:4] - cur_state[2:4])
        delta = goal_state[4] - cur_state[4]
        angle_diff = np.abs(np.arctan2(np.sin(delta), np.cos(delta)))
        success = (pos_diff < 20) and (angle_diff < np.pi / 9)
        state_dist = np.linalg.norm(goal_state - cur_state)
        return {
            "success": success,
            "state_dist": state_dist,
            "block_pos_dist": pos_diff,
            "block_angle_dist_deg": np.degrees(angle_diff),
            "agent_pos_dist": np.linalg.norm(goal_state[:2] - cur_state[:2]),
        }

    def prepare(self, seed, init_state):
        """
        Reset with controlled init_state
        obs: (H W C)
        state: (state_dim)
        """
        self.seed(seed)
        self.reset_to_state = init_state
        obs, state = self.reset()
        return obs, state

    def step_multiple(self, actions):
        """
        infos: dict, each key has shape (T, ...)
        """
        obses = []
        rewards = []
        dones = []
        infos = []
        for action in actions:
            o, r, d, info = self.step(action)
            obses.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        obses = aggregate_dct(obses)
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        infos = aggregate_dct(infos)
        return obses, rewards, dones, infos

    def rollout(self, seed, init_state, actions):
        """
        only returns np arrays of observations and states
        seed: int
        init_state: (state_dim, )
        actions: (T, action_dim)
        obses: dict (T, H, W, C)
        states: (T, D)
        """
        obs, state = self.prepare(seed, init_state)
        obses, rewards, dones, infos = self.step_multiple(actions)
        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])
        states = np.stack(states)
        return obses, states
