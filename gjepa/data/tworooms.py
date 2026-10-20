"""Episode splits and dataset-specific physical-state transforms."""

import json
import time
from pathlib import Path
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback


def episode_split(dataset, seed, fraction, run_dir):
    rng = np.random.default_rng(seed)
    episodes = rng.permutation(len(dataset.lengths))
    n = int(len(episodes) * fraction)
    train_eps, val_eps = episodes[:n], episodes[n:]
    membership = np.zeros(len(episodes), dtype=bool)
    membership[train_eps] = True
    train_ids, val_ids = [], []
    for idx, (ep, _) in enumerate(dataset.clip_indices):
        (train_ids if membership[ep] else val_ids).append(idx)
    rng.shuffle(val_ids)  # Fixed, dispersed prefix when validation is capped.
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        seed=seed,
        train_episodes=train_eps.tolist(),
        val_episodes=val_eps.tolist(),
        train_windows=len(train_ids),
        val_windows=len(val_ids),
        validation_prefix_window_indices=val_ids[:6400],
        state_units="agent xy / 224; fixed coordinate-unit scaling, no fitted z-score",
        dataset=str(dataset.h5_path),
    )
    with (root / "split_manifest.json").open("w") as f:
        json.dump(manifest, f)
    return torch.utils.data.Subset(dataset, train_ids), torch.utils.data.Subset(
        dataset, val_ids
    )


class TimingCallback(Callback):
    def __init__(self, root):
        self.path = Path(root) / "throughput.jsonl"
        self.last_time = None
        self.last_step = 0

    def on_train_start(self, trainer, pl_module):
        self.last_time = time.monotonic()
        self.last_step = trainer.global_step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % 20:
            return
        now = time.monotonic()
        row = dict(
            epoch=trainer.current_epoch,
            step=trainer.global_step,
            seconds_per_update=(now - self.last_time)
            / (trainer.global_step - self.last_step),
            unix_time=time.time(),
        )
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        self.last_time, self.last_step = now, trainer.global_step
