from functools import partial

import torch

from .seg_head import ProgressiveUpHead
from .backbone_utils import ModelWithIntermediateLayers, _get_backbone_out_indices


class FeatureDecoder(torch.nn.Module):
    def __init__(self, segmentation_model: torch.nn.ModuleList, autocast_ctx):
        super().__init__()
        self.segmentation_model = segmentation_model
        self.autocast_ctx = autocast_ctx

    def forward(self, inputs):
        with self.autocast_ctx():
            for module in self.segmentation_model:
                inputs = module.forward(inputs)
        return inputs


def build_segmentation_decoder(
        backbone_model,
        backbone_out_layers="four_even_intervals",
        hidden_dim=2048,
        num_classes=150,
        dropout=0.1,
        tuning=False,
        autocast_dtype=torch.float32,
):
    patch_size = backbone_model.patch_size
    backbone_indices_to_use = _get_backbone_out_indices(backbone_model, backbone_out_layers)
    autocast_ctx = partial(torch.autocast, device_type="cuda", enabled=True, dtype=autocast_dtype)

    backbone_model = ModelWithIntermediateLayers(
        backbone_model,
        n=backbone_indices_to_use,
        autocast_ctx=autocast_ctx,
        reshape=True,
        return_class_token=False,
        tuning=tuning,
    )
    if not tuning:
        backbone_model.requires_grad_(False)
        backbone_model.eval()
    else:
        backbone_model.requires_grad_(True)
        backbone_model.train()

    embed_dim = backbone_model.feature_model.embed_dim
    if isinstance(embed_dim, int):
        if backbone_out_layers in ["four_last", "four_even_intervals"]:
            embed_dim = [embed_dim] * 4
        else:
            embed_dim = [embed_dim]

    decoder = ProgressiveUpHead(
        in_channels=embed_dim,
        n_output_channels=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )

    segmentation_model = FeatureDecoder(
        torch.nn.ModuleList(
            [
                backbone_model,
                decoder,
            ]
        ),
        autocast_ctx=autocast_ctx,
    )
    return segmentation_model
