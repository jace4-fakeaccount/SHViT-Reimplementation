import torch
import torch.nn as nn
from pydantic import BaseModel
from timm.models.registry import register_model

from shvit import SHViT


class ShvitVariantConfig(BaseModel):
    embed_dim: list[int]
    depth: list[int]
    partial_dim: list[int]
    types: list[str]


CONFIG_MAP = {
    "shvit_s1": ShvitVariantConfig(
        embed_dim=[128, 224, 320],
        depth=[2, 4, 5],
        partial_dim=[32, 48, 68],
        types=["i", "s", "s"],
    ),
    "shvit_s2": ShvitVariantConfig(
        embed_dim=[128, 308, 448],
        depth=[2, 4, 5],
        partial_dim=[32, 66, 96],
        types=["i", "s", "s"],
    ),
}


def _create_shvit(
    variant: str,
    pretrained=False,
    num_classes=1000,
    distillation=False,
    fuse=False,
    checkpoint_path=None,
    **kwargs,
):
    if variant not in CONFIG_MAP:
        raise ValueError(f"Unknown variant {variant}")
    config = CONFIG_MAP[variant]
    model = SHViT(
        num_classes=num_classes, distillation=distillation, fuse=fuse, **config.dict()
    )
    if pretrained:
        pretrained = _checkpoint_url_format.format(pretrained)
        checkpoint = torch.hub.load_state_dict_from_url(pretrained, map_location="cpu")
        d = checkpoint["model"]
        D = model.state_dict()
        for k in d.keys():
            if D[k].shape != d[k].shape:
                d[k] = d[k][:, :, None, None]
        model.load_state_dict(d)
    if fuse:
        replace_batchnorm(model)
    return model


def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, "fuse"):
            fused = child.fuse()
            setattr(net, child_name, fused)
            replace_batchnorm(fused)
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)


def register_shvit_models():
    model_names = ["shvit_s1", "shvit_s2"]

    for model_name in model_names:

        def _factory(variant_name=model_name, **kwargs):
            return _create_shvit(variant=variant_name, **kwargs)

        register_model(model_name)(_factory)
        print(f"Registered {model_name} with timm.")


register_shvit_models()
