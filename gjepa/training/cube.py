import hashlib
import json
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from gjepa.modules import SIGReg
from gjepa.invariance import invariant_losses
from gjepa.training.utils import (
    get_column_normalizer,
    get_img_preprocessor,
    SaveCkptCallback,
)
from gjepa.data.cube import (
    episode_split,
    TimingCallback,
    StopAfterEpochCallback,
    compact_state_transform,
    train_action_transform,
)


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    batch["state"] = batch["state"].float()

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]  # label
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]
    if cfg.loss.get("lambda_state", 0.0) > 0:
        _s_hist = self.model.state_head(ctx_emb)
        output["L_state"] = (_s_hist - batch["state"][:, :ctx_len]).pow(2).mean()
        output["loss"] += cfg.loss.lambda_state * output["L_state"]
    if cfg.loss.get("lambda_pred_state", 0.0) > 0:
        _s_fut = self.model.state_head(pred_emb)
        output["L_pred_state"] = (_s_fut - batch["state"][:, n_preds:]).pow(2).mean()
        output["loss"] += cfg.loss.lambda_pred_state * output["L_pred_state"]

    for key, value in invariant_losses(self, batch, emb, pred_emb, cfg).items():
        output[key] = value
        output["loss"] += cfg.loss["lambda_" + key[2:]] * value
    _loss_keys = {"L_state", "L_pred_state", "L_inv", "L_inv_trans"}
    losses_dict = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items()
        if "loss" in k or k in _loss_keys
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parents[2] / "configs/train"),
    config_name="cube",
)
def run(cfg):
    from gjepa.training.utils import prepare_run

    prepare_run(cfg)
    pl.seed_everything(cfg.seed, workers=True)
    run_dir = Path(cfg.output_dir).resolve()
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    dataset = swm.data.HDF5Dataset(path=dataset_name, transform=None, **dataset_cfg)
    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    if cfg.get("episode_holdout", False):
        train_set, val_set, train_eps = episode_split(
            dataset, cfg.seed, cfg.train_split, run_dir
        )
        transforms = [
            get_img_preprocessor(
                source="pixels", target="pixels", img_size=cfg.img_size
            ),
            train_action_transform(dataset, train_eps, run_dir),
            compact_state_transform(dataset, train_eps, run_dir),
        ]
    else:
        transforms = [
            get_img_preprocessor(
                source="pixels", target="pixels", img_size=cfg.img_size
            )
        ]
        with open_dict(cfg):
            for col in cfg.data.dataset.keys_to_load:
                if not col.startswith("pixels"):
                    transforms.append(get_column_normalizer(dataset, col, col))
        train_set, val_set = spt.data.random_split(
            dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
        )
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim("action")
        )
    dataset.transform = spt.data.transforms.Compose(*transforms)

    # Fail before training if chunk standardization or compact-state selection
    # is not actually applied by the dataset transform.
    probe = train_set[0]
    if tuple(probe["action"].shape) != (cfg.history_size + cfg.num_preds, 25):
        raise ValueError(f"unexpected transformed action shape {probe['action'].shape}")
    if tuple(probe["state"].shape) != (cfg.history_size + cfg.num_preds, 18):
        raise ValueError(f"unexpected compact state shape {probe['state'].shape}")
    if (
        not torch.isfinite(probe["action"]).all()
        or not torch.isfinite(probe["state"]).all()
    ):
        raise ValueError("non-finite transformed action/state probe")
    with (run_dir / "data_summary.json").open("w") as file:
        json.dump(
            {
                "action_shape": list(probe["action"].shape),
                "state_shape": list(probe["state"].shape),
                "action_min": float(probe["action"].min()),
                "action_max": float(probe["action"].max()),
                "state_min": float(probe["state"].min()),
                "state_max": float(probe["state"].max()),
                "compact_state_indices": [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    26,
                    27,
                ],
            },
            file,
            indent=2,
        )

    # ``merge_col`` opens the HDF5 file while constructing the dataset.  Close
    # that parent-process handle before forking DataLoader workers so every
    # worker lazily opens its own SWMR read handle.  This preserves the shared
    # in-memory state cache while avoiding an inherited h5py handle.
    if getattr(dataset, "h5_file", None) is not None:
        dataset.h5_file.close()
        dataset.h5_file = None

    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False
    )

    ##############################
    ##       model / optim      ##
    ##############################

    # Persist the head shape in config.json so native load_pretrained can rebuild it.
    with open_dict(cfg):
        cfg.model.state_dim = 18
        cfg.model.state_embed_dim = cfg.embed_dim
    world_model = hydra.utils.instantiate(cfg.model)
    digest = hashlib.sha256()
    for name, tensor in sorted(world_model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    with (run_dir / "initial_model_sha256.txt").open("w") as file:
        file.write(digest.hexdigest() + "\n")

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_dir = Path(cfg.output_dir).resolve()

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_dir=run_dir,
        cfg=cfg.model,
        epoch_interval=1,
    )

    callbacks = [object_dump_callback, TimingCallback(run_dir)]
    if cfg.get("stop_after_epochs") is not None:
        callbacks.append(StopAfterEpochCallback(cfg.stop_after_epochs))
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / "checkpoints" / "model_latest.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
        seed=cfg.seed,
    )

    manager()
    return


if __name__ == "__main__":
    run()
