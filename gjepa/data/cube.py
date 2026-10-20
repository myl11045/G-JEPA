"""Episode splits and dataset-specific physical-state transforms."""

import hashlib
import json
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback


COMPACT_STATE_INDICES = (
    0,
    1,
    2,
    3,
    4,
    5,  # arm joint positions
    12,
    13,
    14,
    15,
    16,
    17,
    18,  # end-effector pose, gripper, contact
    19,
    20,
    21,
    26,
    27,  # cube position and yaw (cos, sin)
)


class ChunkActionStandardizer:
    """Apply one 5-D train-only scaler to every action in a flattened chunk."""

    def __init__(self, mean, scale):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.scale = torch.as_tensor(scale, dtype=torch.float32)

    def __call__(self, value):
        value = value.float()
        if value.shape[-1] % 5:
            raise ValueError(f"action width must be divisible by 5, got {value.shape}")
        shape = value.shape
        actions = value.reshape(*shape[:-1], -1, 5)
        actions = (actions - self.mean.to(actions)) / self.scale.to(actions)
        return actions.reshape(shape)


class CompactInteractionState:
    """Select and standardize the 18-D robot-object interaction state."""

    def __init__(self, mean, scale):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.scale = torch.as_tensor(scale, dtype=torch.float32)

    def __call__(self, value):
        if value.shape[-1] != 28:
            raise ValueError(
                f"expected OGBench observation width 28, got {value.shape}"
            )
        index = torch.as_tensor(COMPACT_STATE_INDICES, device=value.device)
        selected = value.index_select(-1, index).float()
        return (selected - self.mean.to(selected)) / self.scale.to(selected)


def compact_state_transform(dataset, train_eps, run_dir):
    """Fit a train-episode-only z-score scaler for the selected 18-D state."""
    with h5py.File(dataset.h5_path, "r") as handle:
        # The official H5 stores the 28-D vector as ``observation``;
        # the dataset config later exposes it to the model as ``state``.
        states = np.asarray(handle["observation"][:], dtype=np.float64)
        offsets = np.asarray(handle["ep_offset"][:], dtype=np.int64)
        lengths = np.asarray(handle["ep_len"][:], dtype=np.int64)
    selected_rows = np.zeros(len(states), dtype=bool)
    for episode in np.asarray(train_eps, dtype=np.int64):
        start = int(offsets[episode])
        selected_rows[start : start + int(lengths[episode])] = True
    train_states = states[selected_rows][:, COMPACT_STATE_INDICES]
    finite = np.isfinite(train_states).all(axis=1)
    train_states = train_states[finite]
    if train_states.ndim != 2 or train_states.shape[1] != len(COMPACT_STATE_INDICES):
        raise ValueError(f"unexpected compact train state shape {train_states.shape}")
    mean = train_states.mean(axis=0)
    raw_scale = train_states.std(axis=0, ddof=0)
    scale = np.maximum(raw_scale, 1e-6)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError(f"invalid state scaler: mean={mean}, scale={scale}")
    probe = train_states[:4096]
    roundtrip = ((probe - mean) / scale) * scale + mean
    payload = {
        "kind": "train-only population StandardScaler",
        "source_dim": 28,
        "output_dim": len(COMPACT_STATE_INDICES),
        "indices": list(COMPACT_STATE_INDICES),
        "components": [
            "joint_pos_0",
            "joint_pos_1",
            "joint_pos_2",
            "joint_pos_3",
            "joint_pos_4",
            "joint_pos_5",
            "ee_x",
            "ee_y",
            "ee_z",
            "ee_yaw_cos",
            "ee_yaw_sin",
            "gripper_opening",
            "gripper_contact",
            "cube_x",
            "cube_y",
            "cube_z",
            "cube_yaw_cos",
            "cube_yaw_sin",
        ],
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "raw_scale": raw_scale.tolist(),
        "n_samples": int(len(train_states)),
        "n_nonfinite_rows_removed": int(selected_rows.sum() - len(train_states)),
        "train_episode_sha256": hashlib.sha256(
            np.asarray(train_eps, dtype=np.int64).tobytes()
        ).hexdigest(),
        "max_probe_roundtrip_abs_error": float(np.max(np.abs(roundtrip - probe))),
    }
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "state_scaler.json").open("w") as file:
        json.dump(payload, file, indent=2)
    return dt.transforms.WrapTorchTransform(
        CompactInteractionState(mean, scale), source="state", target="state"
    )


def train_action_transform(dataset, train_eps, run_dir):
    """Fit a population StandardScaler on raw 5-D actions from train episodes."""
    with h5py.File(dataset.h5_path, "r") as handle:
        actions = np.asarray(handle["action"][:], dtype=np.float64)
        offsets = np.asarray(handle["ep_offset"][:], dtype=np.int64)
        lengths = np.asarray(handle["ep_len"][:], dtype=np.int64)
    selected = np.zeros(len(actions), dtype=bool)
    for episode in np.asarray(train_eps, dtype=np.int64):
        start = int(offsets[episode])
        selected[start : start + int(lengths[episode])] = True
    train_actions = actions[selected]
    n_selected = int(len(train_actions))
    finite = np.isfinite(train_actions).all(axis=1)
    train_actions = train_actions[finite]
    if train_actions.ndim != 2 or train_actions.shape[1] != 5:
        raise ValueError(f"unexpected train action shape {train_actions.shape}")
    mean = train_actions.mean(axis=0)
    scale = train_actions.std(axis=0, ddof=0)
    if (
        not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
    ):
        raise ValueError(f"invalid action scaler: mean={mean}, scale={scale}")
    probe = train_actions[:4096]
    roundtrip = ((probe - mean) / scale) * scale + mean
    payload = {
        "kind": "train-only population StandardScaler",
        "source_dim": 5,
        "chunk_rule": "standardize each raw 5-D action, then flatten frameskip=5 to 25-D",
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "n_samples": int(len(train_actions)),
        "n_nonfinite_rows_removed": n_selected - int(len(train_actions)),
        "train_episode_sha256": hashlib.sha256(
            np.asarray(train_eps, dtype=np.int64).tobytes()
        ).hexdigest(),
        "max_probe_roundtrip_abs_error": float(np.max(np.abs(roundtrip - probe))),
    }
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "action_scaler.json").open("w") as file:
        json.dump(payload, file, indent=2)
    return dt.transforms.WrapTorchTransform(
        ChunkActionStandardizer(mean, scale), source="action", target="action"
    )


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
        state_definition="compact18 observation indices " + str(COMPACT_STATE_INDICES),
        state_units="train-episode-only per-dimension population z-score",
        dataset=str(dataset.h5_path),
    )
    with (root / "split_manifest.json").open("w") as f:
        json.dump(manifest, f)
    return (
        torch.utils.data.Subset(dataset, train_ids),
        torch.utils.data.Subset(dataset, val_ids),
        train_eps,
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


class StopAfterEpochCallback(Callback):
    """Stop cleanly after N completed epochs while preserving the full scheduler."""

    def __init__(self, epochs):
        self.epochs = int(epochs)
        if self.epochs < 1:
            raise ValueError("epochs must be positive")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if trainer.current_epoch + 1 >= self.epochs:
            trainer.should_stop = True
