"""Generate an action plan for a current/goal image pair from an HDF5 episode."""

import argparse
from pathlib import Path

from gjepa.control import add_planning_arguments, planner_from_args, read_task
from gjepa.io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_planning_arguments(parser)
    parser.add_argument("--output", type=Path, default=Path("runs/plan.json"))
    args = parser.parse_args()
    task = read_task(args.data, args.episode, args.start, args.goal_offset)
    actions, costs = planner_from_args(args).plan(
        task["current_image"], task["goal_image"]
    )
    write_json(
        args.output,
        {
            "actions": actions.tolist(),
            "elite_cost_per_iteration": costs,
            "episode": args.episode,
            "start": args.start,
            "goal_offset": args.goal_offset,
            "mode": args.mode,
        },
    )
    print(f"Saved {len(actions)} proposed actions to {args.output}")


if __name__ == "__main__":
    main()
