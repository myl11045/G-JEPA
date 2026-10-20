from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from gjepa.modules import SIGReg, build_state_decoder
from gjepa.training.utils import (
    get_column_normalizer,
    get_img_preprocessor,
    SaveCkptCallback,
)
from gjepa.invariance import invariant_losses


class StopAfterEpochCallback(pl.Callback):
    """Stop a 20-epoch-scheduled run after a completed epoch."""

    def __init__(self, stop_after_epochs):
        super().__init__()
        self.stop_after_epochs = int(stop_after_epochs)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch + 1 >= self.stop_after_epochs:
            trainer.should_stop = True
            print(f"STOP_AFTER_EPOCH={trainer.current_epoch + 1}", flush=True)


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

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
        target_weight = float(cfg.loss.lambda_pred_state)
        warmup_updates = int(cfg.loss.get("pred_state_warmup_updates", 0))
        if warmup_updates > 0:
            progress = min(float(self.global_step) / float(warmup_updates), 1.0)
            pred_state_weight = target_weight * progress
        else:
            pred_state_weight = target_weight
        output["lambda_pred_state"] = output["loss"].new_tensor(pred_state_weight)
        output["loss"] += pred_state_weight * output["L_pred_state"]

    for key, value in invariant_losses(self, batch, emb, pred_emb, cfg).items():
        output[key] = value
        output["loss"] += float(cfg.loss["lambda_" + key[2:]]) * value

    _loss_keys = {
        "L_state",
        "L_pred_state",
        "lambda_pred_state",
        "L_inv",
        "L_inv_trans",
    }
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
    config_name="pusht",
)
def run(cfg):
    from gjepa.training.utils import prepare_run

    prepare_run(cfg)
    # Initialization, split, and training seeds are configured separately.
    init_seed = int(cfg.get("init_seed", cfg.seed))
    train_seed = int(cfg.get("train_seed", cfg.seed))
    pl.seed_everything(init_seed, workers=True)
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    dataset = swm.data.HDF5Dataset(path=dataset_name, transform=None, **dataset_cfg)
    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col, cfg.output_dir)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim("action")
        )

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False
    )

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)
    if (
        cfg.loss.get("lambda_state", 0.0) > 0
        or cfg.loss.get("lambda_pred_state", 0.0) > 0
    ):
        world_model.state_head = build_state_decoder(
            kind=cfg.observer.type,
            d_in=cfg.embed_dim,
            d_out=dataset.get_dim("state"),
        )
        observer_params = sum(p.numel() for p in world_model.state_head.parameters())
        print(
            f"Temporal module: {cfg.observer.type} "
            f"params={observer_params} d_in={cfg.embed_dim} "
            f"d_out={dataset.get_dim('state')}",
            flush=True,
        )

    with open_dict(cfg):
        cfg.model.observer_type = cfg.observer.type

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
    stop_callback = StopAfterEpochCallback(cfg.stop_after_epochs)

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback, stop_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / "checkpoints" / "model_latest.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
        seed=train_seed,
    )

    manager()
    return


if __name__ == "__main__":
    run()
