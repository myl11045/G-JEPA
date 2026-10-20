"""Train the conditional action prior from a frozen latent cache."""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from gjepa.action_prior import DiffusionActor
from gjepa.io import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=2048)
    a = p.parse_args()
    if not (a.cache / "COMPLETE.json").exists():
        raise ValueError("Feature extraction is incomplete")
    a.output.mkdir(parents=True, exist_ok=False)
    meta = json.loads((a.cache / "metadata.json").read_text())
    ids, lengths, offsets, raw, z = [
        np.load(a.cache / (name + ".npy"))
        for name in ["episode_ids", "ep_lengths", "ep_offsets", "actions", "embeddings"]
    ]
    train_ids = set(meta["train_episode_ids"])
    bad = np.r_[0, np.cumsum(~np.isfinite(raw).all(-1))]
    train_rows, val_rows, train_frames = [], [], []
    for ep, n, start in zip(ids, lengths, offsets):
        rows = np.arange(start, start + n - 25, dtype=np.int64)
        rows = rows[bad[rows + 25] == bad[rows]]
        (train_rows if ep in train_ids else val_rows).append(rows)
        if ep in train_ids:
            train_frames.append(np.arange(start, start + n))
    tr, vr = [
        torch.from_numpy(np.concatenate(rows)).to(a.device)
        for rows in [train_rows, val_rows]
    ]
    if not len(tr) or not len(vr):
        raise ValueError("Empty prior training or validation split")
    torch.manual_seed(3072)
    Z = torch.from_numpy(z).to(a.device)
    mean = np.array(meta["action_mean"], np.float32)
    scale = np.array(meta["action_scale"], np.float32)
    A = torch.from_numpy((raw - mean) / scale).to(a.device)
    model = DiffusionActor(action_dim=raw.shape[1]).to(a.device)
    with torch.no_grad():
        features = Z[torch.from_numpy(np.concatenate(train_frames)).to(a.device)]
        model.z_mean.copy_(features.mean(0))
        model.z_std.copy_(features.std(0, unbiased=False).clamp_min(1e-3))
        for bound in ["low", "high"]:
            value = np.tile(
                (np.asarray(meta["action_" + bound], np.float32) - mean) / scale, 25
            )
            getattr(model, "action_" + bound).copy_(
                torch.from_numpy(value).to(a.device)
            )
        del features
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    total = math.ceil(len(tr) / a.batch) * a.epochs
    warm = min(1000, max(1, total // 10))

    def schedule(step):
        if step < warm:
            return (step + 1) / warm
        return 0.01 + 0.99 * 0.5 * (
            1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm)))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    generator = torch.Generator(device=a.device).manual_seed(3072)
    action_steps = torch.arange(25, device=a.device)
    meta.update(
        epochs=a.epochs,
        batch=a.batch,
        train_rows=len(tr),
        val_rows=len(vr),
        seed=3072,
        ema_decay=0.995,
        lr=1e-4,
        weight_decay=1e-4,
        model_config=model.config,
    )
    write_json(a.output / "config.json", meta)
    updates, best = 0, float("inf")

    def save(name, epoch, val_loss):
        # Weights and scalar metadata are loadable with weights_only=True.
        torch.save(
            {
                "state_dict": ema.state_dict(),
                "config": model.config,
                "metadata": meta,
                "epoch": epoch,
                "val_loss": val_loss,
                "global_step": updates,
            },
            a.output / name,
        )

    for epoch in range(1, a.epochs + 1):
        model.train()
        loss_sum = 0.0
        for select in torch.randperm(
            len(tr), device=a.device, generator=generator
        ).split(a.batch):
            rows = tr[select]
            targets = A[rows[:, None] + action_steps].reshape(len(rows), -1)
            t = torch.randint(100, (len(rows),), device=a.device, generator=generator)
            epsilon = torch.randn(targets.shape, device=a.device, generator=generator)
            prediction = model(
                model.noise(targets, t, epsilon), Z[rows], Z[rows + 25], t
            )
            loss = (prediction - epsilon).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite prior loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            updates += 1
            with torch.no_grad():
                for target, source in zip(ema.parameters(), model.parameters()):
                    target.lerp_(source, 0.005)
            loss_sum += float(loss.detach()) * len(rows)
        vg = torch.Generator(device=a.device).manual_seed(20260912)
        val_sum = 0.0
        with torch.no_grad():
            for rows in vr.split(a.batch):
                targets = A[rows[:, None] + action_steps].reshape(len(rows), -1)
                t = torch.randint(100, (len(rows),), device=a.device, generator=vg)
                epsilon = torch.randn(targets.shape, device=a.device, generator=vg)
                pred = ema(ema.noise(targets, t, epsilon), Z[rows], Z[rows + 25], t)
                val_sum += float((pred - epsilon).square().mean()) * len(rows)
        vl = val_sum / len(vr)
        row = dict(
            epoch=epoch,
            global_step=updates,
            train_epsilon_mse=loss_sum / len(tr),
            val_epsilon_mse=vl,
        )
        with (a.output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(row, flush=True)
        if vl < best:
            best = vl
            save("best.pt", epoch, vl)
        save("final.pt", epoch, vl)
    write_json(
        a.output / "COMPLETE.json",
        {"global_step": updates, "best_val_epsilon_mse": best},
    )


if __name__ == "__main__":
    main()
