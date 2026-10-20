"""Execute image-conditioned plans in PushT or OGBench Cube and save a rollout."""

import argparse
import json
from pathlib import Path

import numpy as np

from gjepa.control import add_planning_arguments, planner_from_args, read_task
from gjepa.io import write_json
from gjepa.simulation import CubeSimulation, PushTSimulation, run_rollout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_planning_arguments(parser)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument(
        "--execute",
        type=int,
        default=5,
        help="actions executed before replanning (1-25)",
    )
    parser.add_argument("--stop-on-success", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/rollout"))
    args = parser.parse_args()
    if (args.output / "rollout.npz").exists():
        parser.error("Choose a new output directory to keep the existing rollout")
    task = read_task(
        args.data, args.episode, args.start, args.goal_offset, simulation=True
    )
    planner = planner_from_args(args)
    env = (PushTSimulation if args.domain == "pusht" else CubeSimulation)(
        task, args.seed
    )
    try:
        arrays, summary = run_rollout(
            env,
            planner,
            task["goal_image"],
            args.max_steps,
            args.execute,
            args.stop_on_success,
        )
    finally:
        env.close()
    metadata = {
        "domain": args.domain,
        "mode": args.mode,
        "episode": args.episode,
        "start": args.start,
        "goal_offset": args.goal_offset,
        "position_unit": "px" if args.domain == "pusht" else "cm",
        "execute": args.execute,
        "max_steps": args.max_steps,
        "cost": args.cost,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "rollout.npz", **arrays, metadata=json.dumps(metadata)
    )
    write_json(args.output / "summary.json", {**metadata, **summary})
    print(f"Saved {summary['environment_steps']} executed steps to {args.output}")


if __name__ == "__main__":
    main()
