"""Portable configuration, checkpoint loading, and image preprocessing."""

import hashlib
import json
from pathlib import Path

import hydra
import numpy as np
import torch
from torchvision.transforms import v2 as T


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def portable_config(config):
    if isinstance(config, dict):
        result = {k: portable_config(v) for k, v in config.items()}
        target = result.get("_target_", "")
        if target == "jepa.JEPA":
            result["_target_"] = "gjepa.world_model.JEPA"
        elif target.startswith("module."):
            result["_target_"] = "gjepa.modules." + target.split(".", 1)[1]
        elif target == "stable_pretraining.backbone.utils.vit_hf":
            result["_target_"] = "gjepa.backbone.vit_hf"
        return result
    if isinstance(config, list):
        return [portable_config(x) for x in config]
    return config


def load_world(path, device="cpu"):
    path = Path(path)
    config = portable_config(json.loads(path.with_name("config.json").read_text()))
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = hydra.utils.instantiate(config)
    if not any(k.startswith("state_head.") for k in state):
        del model.state_head
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


def load_prior(path, device="cpu", world_path=None):
    from .action_prior import DiffusionActor

    saved = torch.load(path, map_location="cpu", weights_only=True)
    config = dict(saved["config"])
    config.setdefault("action_dim", saved["state_dict"]["inp.weight"].shape[1] // 25)
    model = DiffusionActor(**config)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.metadata = saved.get("metadata", {})
    expected = saved.get("metadata", {}).get("world_checkpoint_sha")
    if world_path is not None and expected and sha256(world_path) != expected:
        raise ValueError(
            "The action prior was trained with a different world checkpoint."
        )
    return model.to(device).eval().requires_grad_(False)


def image_transform():
    return T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            T.Resize(224),
        ]
    )


def image_tensor(image, device):
    return (
        image_transform()(np.asarray(image).copy()).unsqueeze(0).unsqueeze(0).to(device)
    )
