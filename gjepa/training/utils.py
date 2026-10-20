"""Training transforms, metadata, and resumable checkpoints."""

import platform
from pathlib import Path
import importlib.metadata

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf, open_dict
from stable_pretraining import data as dt

from gjepa.io import write_json


def prepare_run(cfg):
    root = Path(cfg.output_dir).resolve()
    if (root / "config.yaml").exists() and not cfg.resume:
        raise FileExistsError(
            f"{root} already contains a run. Use resume=true or a new output_dir."
        )
    root.mkdir(parents=True, exist_ok=True)
    with open_dict(cfg):
        if int(cfg.loader.num_workers) == 0:
            cfg.loader.persistent_workers = False
            cfg.loader.pop("prefetch_factor", None)
        elif "prefetch_factor" not in cfg.loader:
            cfg.loader.prefetch_factor = 3
    # stable-pretraining 0.1.6 uses the pre-0.20 torchvision Transform interface.
    cls = dt.transforms.Transform
    if not getattr(cls, "_gjepa_compat", False):
        original = cls.__getattr__

        def compat(self, name):
            if name == "transform":
                return self._transform
            return original(self, name)

        cls.__getattr__ = compat
        cls._gjepa_compat = True
    packages = [
        "torch",
        "torchvision",
        "transformers",
        "stable-pretraining",
        "stable-worldmodel",
        "lightning",
        "numpy",
        "h5py",
    ]
    versions = {name: importlib.metadata.version(name) for name in packages}
    write_json(
        root / "environment.json",
        {"python": platform.python_version(), "packages": versions},
    )


def get_img_preprocessor(source, target, img_size=224):
    return dt.transforms.Compose(
        dt.transforms.ToImage(
            **dt.dataset_stats.ImageNet, source=source, target=target
        ),
        dt.transforms.Resize(img_size, source=source, target=target),
    )


class ZScoreNormalizer:
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, value):
        return ((value - self.mean) / self.std).float()


def get_column_normalizer(dataset, source, target, run_dir=None):
    data = torch.from_numpy(np.array(dataset.get_col_data(source)))
    data = data[~torch.isnan(data).any(dim=1)]
    mean, std = data.mean(0, keepdim=True), data.std(0, keepdim=True)
    if run_dir is not None:
        write_json(
            Path(run_dir) / f"{source}_scaler.json",
            {"mean": mean[0].tolist(), "scale": std[0].tolist()},
        )
    return dt.transforms.WrapTorchTransform(
        ZScoreNormalizer(mean, std), source=source, target=target
    )


class SaveCkptCallback(Callback):
    def __init__(self, run_dir, cfg, epoch_interval=1):
        super().__init__()
        self.root, self.cfg, self.interval = Path(run_dir), cfg, epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if not trainer.is_global_zero or (
            epoch % self.interval and epoch != trainer.max_epochs
        ):
            return
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(
            self.root / "config.json", OmegaConf.to_container(self.cfg, resolve=True)
        )
        torch.save(
            pl_module.model.state_dict(), self.root / f"weights_epoch_{epoch}.pt"
        )
        folder = self.root / "checkpoints"
        folder.mkdir(exist_ok=True)
        trainer.save_checkpoint(str(folder / "model_latest.ckpt"))
