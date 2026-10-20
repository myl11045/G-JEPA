"""Cache frozen world-model embeddings for action-prior training."""

import argparse
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from gjepa.io import load_world, image_transform, sha256, write_json


class Frames(Dataset):
    def __init__(self, path, rows):
        self.path, self.rows, self.handle = str(path), rows, None
        self.transform = image_transform()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self.handle is None:
            self.handle = h5py.File(self.path, "r", swmr=True)
        return index, self.transform(self.handle["pixels"][int(self.rows[index])])


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=["pusht", "cube"], required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--world", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    with h5py.File(a.data, "r") as f:
        ids = np.arange(len(f["ep_len"]), dtype=np.int64)
        if a.domain == "cube":
            split = json.loads(a.world.with_name("split_manifest.json").read_text())
            ordered = np.asarray(sorted(split["train_episodes"]), dtype=np.int64)
            # Preserve the original two-shard concatenation order for training RNG.
            ids = np.concatenate([ordered[::2], ordered[1::2]])
        lengths = np.asarray(f["ep_len"])[ids].astype(np.int64)
        source_offsets = np.asarray(f["ep_offset"])[ids]
        offsets = np.r_[0, np.cumsum(lengths)[:-1]].astype(np.int64)
        rows = np.concatenate(
            [np.arange(start, start + n) for start, n in zip(source_offsets, lengths)]
        )
        raw = np.asarray(f["action"])[rows]
    for name, value in [
        ("episode_ids", ids),
        ("ep_lengths", lengths),
        ("ep_offsets", offsets),
        ("actions", raw),
    ]:
        np.save(a.output / (name + ".npy"), value)
    order = np.random.default_rng(20260912).permutation(np.sort(ids))
    nval = int(np.ceil(0.1 * len(ids))) if a.domain == "pusht" else 500
    val_ids, train_ids = order[:nval], order[nval:]
    if not len(train_ids):
        raise ValueError("Not enough episodes for the action-prior split")
    if a.domain == "pusht":
        mean, scale = [-0.0087, 0.0068], [0.2019, 0.2002]
        train_raw = np.concatenate(
            [
                raw[o : o + n]
                for ep, n, o in zip(ids, lengths, offsets)
                if ep in set(train_ids)
            ]
        )
        if not np.isfinite(train_raw).all():
            raise ValueError("Nonfinite PushT actions in prior training episodes")
        low, high = train_raw.min(0).tolist(), train_raw.max(0).tolist()
    else:
        scaler = json.loads(a.world.with_name("action_scaler.json").read_text())
        mean, scale, low, high = scaler["mean"], scaler["scale"], [-1.0] * 5, [1.0] * 5
    meta = dict(
        domain=a.domain,
        world_checkpoint_sha=sha256(a.world),
        train_episode_ids=sorted(train_ids.tolist()),
        val_episode_ids=sorted(val_ids.tolist()),
        action_mean=mean,
        action_scale=scale,
        action_low=low,
        action_high=high,
        goal_offset=25,
        split_seed=20260912,
        frames=len(rows),
    )
    write_json(a.output / "metadata.json", meta)
    model = load_world(a.world, a.device)
    torch.backends.cudnn.allow_tf32 = False
    loader = DataLoader(
        Frames(a.data, rows),
        batch_size=a.batch,
        num_workers=a.workers,
        shuffle=False,
        pin_memory=str(a.device).startswith("cuda"),
    )
    cache = np.lib.format.open_memmap(
        a.output / "embeddings.npy", mode="w+", dtype=np.float32, shape=(len(rows), 192)
    )
    count = 0
    for indices, pixels in loader:
        z = model.encode({"pixels": pixels.to(a.device)[:, None]})["emb"][:, 0]
        if not torch.isfinite(z).all():
            raise ValueError("Nonfinite encoded features")
        cache[indices.numpy()] = z.cpu().numpy()
        count += len(indices)
        if count % (a.batch * 100) == 0:
            print(f"{count}/{len(rows)} frames", flush=True)
    cache.flush()
    write_json(
        a.output / "COMPLETE.json",
        {"frames": count, "world_checkpoint_sha": meta["world_checkpoint_sha"]},
    )


if __name__ == "__main__":
    main()
