"""Appearance-pair invariance losses for the LeWM PushT loss matrix."""

import torch
from torchvision.transforms import functional as VF


@torch.no_grad()
def appearance_pair(pixels):
    """Apply one temporally consistent appearance perturbation per clip."""
    mean = pixels.new_tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    std = pixels.new_tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    raw = (pixels * std + mean).clamp(0, 1)
    clips = []
    for clip in raw:
        if torch.rand((), device=clip.device) < 0.9:
            factors = (0.5 + torch.rand(3, device=clip.device)).tolist()
            clip = VF.adjust_brightness(clip, factors[0])
            clip = VF.adjust_contrast(clip, factors[1])
            clip = VF.adjust_saturation(clip, factors[2])
            clip = (clip + torch.randn_like(clip) * 0.05).clamp(0, 1)
        clips.append(clip)
    return (torch.stack(clips) - mean) / std


def invariant_losses(module, batch, emb, pred_emb, cfg):
    lambda_inv = float(cfg.loss.get("lambda_inv", 0.0))
    lambda_inv_trans = float(cfg.loss.get("lambda_inv_trans", 0.0))
    if not (lambda_inv or lambda_inv_trans):
        return {}

    # The target branch is detached. Fork its RNG and restore model buffers so
    # the extra view does not alter the main dropout/SIGReg stream or BN state.
    buffers = {name: value.clone() for name, value in module.model.named_buffers()}
    devices = [emb.device.index] if emb.is_cuda else []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(int(cfg.seed) + 1000003 + int(module.global_step))
            paired = module.model.encode(
                {
                    "pixels": appearance_pair(batch["pixels"]),
                    "action": batch["action"],
                }
            )
            paired_future = None
            if lambda_inv_trans:
                paired_future = module.model.predict(
                    paired["emb"][:, : cfg.history_size],
                    paired["act_emb"][:, : cfg.history_size],
                )
    finally:
        for name, value in buffers.items():
            parent, _, leaf = name.rpartition(".")
            owner = module.model.get_submodule(parent) if parent else module.model
            owner._buffers[leaf] = value

    result = {}
    if lambda_inv:
        result["L_inv"] = (
            1
            - torch.nn.functional.cosine_similarity(emb, paired["emb"].detach(), dim=-1)
        ).mean()
    if lambda_inv_trans:
        result["L_inv_trans"] = (
            1
            - torch.nn.functional.cosine_similarity(
                pred_emb, paired_future.detach(), dim=-1
            )
        ).mean()
    return result
