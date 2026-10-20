"""Train a G-JEPA world model on image-based offline trajectories."""

import argparse
import importlib
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain", choices=["pusht", "cube", "tworooms"], required=True
    )
    args, overrides = parser.parse_known_args()
    sys.argv = [sys.argv[0], *overrides]
    importlib.import_module(f"gjepa.training.{args.domain}").run()
