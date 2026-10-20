"""ViT configuration used by LeWM (see LICENSE for attribution)."""

from transformers import ViTConfig, ViTModel


def vit_hf(
    size="tiny",
    patch_size=14,
    image_size=224,
    pretrained=False,
    use_mask_token=False,
    **kwargs,
):
    if pretrained:
        raise ValueError("The released world models are trained from scratch.")
    sizes = {"tiny": (192, 12, 3), "small": (384, 12, 6), "base": (768, 12, 12)}
    width, depth, heads = sizes[size]
    config = ViTConfig(
        hidden_size=width,
        num_hidden_layers=depth,
        num_attention_heads=heads,
        intermediate_size=width * 4,
        image_size=image_size,
        patch_size=patch_size,
        **kwargs,
    )
    model = ViTModel(config, add_pooling_layer=False, use_mask_token=use_mask_token)
    model.config.interpolate_pos_encoding = True
    return model
