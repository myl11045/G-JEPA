import numpy as np

from gjepa.simulation import run_rollout


class Environment:
    def __init__(self):
        self.state = np.array([0.0])
        self.goal_state = np.array([5.0])
        self.image = np.zeros((2, 2, 3), dtype=np.uint8)

    def step(self, action):
        self.state += action
        self.image[:] = int(self.state[0])

    def metrics(self):
        distance = float(abs(self.state[0] - self.goal_state[0]))
        return distance, 0.0, distance == 0


class Planner:
    def __init__(self):
        self.conditions = []

    def plan(self, image, goal, round_index):
        self.conditions.append((image.copy(), goal.copy(), round_index))
        return np.ones((25, 1)), []


def test_replanning_uses_new_observation_and_preserves_goal():
    planner, env = Planner(), Environment()
    goal = np.full((2, 2, 3), 5, dtype=np.uint8)
    arrays, summary = run_rollout(env, planner, goal, max_steps=7, execute=3)
    assert [int(c[0][0, 0, 0]) for c in planner.conditions] == [0, 3, 6]
    assert all(np.array_equal(c[1], goal) for c in planner.conditions)
    assert len(arrays["frames"]) == 8 and len(arrays["actions"]) == 7
    assert summary["reached_goal"] and not summary["final_success"]


def test_success_stops_inside_action_prefix():
    arrays, summary = run_rollout(
        Environment(),
        Planner(),
        np.zeros((2, 2, 3)),
        max_steps=20,
        execute=3,
        stop_on_success=True,
    )
    assert len(arrays["actions"]) == 5
    assert summary["environment_steps"] == 5 and summary["final_success"]
    assert len(summary["replans"]) == 2
