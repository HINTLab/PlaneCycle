import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial


def _get_backbone_out_indices(
        model: torch.nn.Module,
        backbone_out_layers: str = "four_even_intervals",
):
    """
    Get indices for output layers of the ViT backbone. For now there are 3 options available:
    last : only extract the last layer, used in segmentation tasks with a bn head.
    four_last: extract the four last layer, used in segmentation tasks with a bn head.
    four_even_intervals : extract outputs every 1/4 of the total number of blocks.
    """
    n_blocks = getattr(model, "n_blocks", 1)
    if backbone_out_layers == "last":
        out_indices = [n_blocks - 1]
    elif backbone_out_layers == "four_last":
        out_indices = [i for i in range(n_blocks - 4, n_blocks)]
    elif backbone_out_layers == "four_even_intervals":
        # Take indices that were used in the paper (for ViT/L only)
        if n_blocks == 24:
            out_indices = [4, 11, 17, 23]
        else:
            out_indices = [i * (n_blocks // 4) - 1 for i in range(1, 5)]
    else:
        raise ValueError(f"Invalid backbone_out_layers: {backbone_out_layers}")
    assert all([out_index < n_blocks for out_index in out_indices])
    return out_indices


class ModelWithIntermediateLayers(nn.Module):
    def __init__(self, feature_model, n, autocast_ctx, reshape=False, return_class_token=True, tuning=False):
        super().__init__()

        self.feature_model = feature_model
        self.tuning = tuning
        if not self.tuning:
            self.feature_model.eval()
        else:
            self.feature_model.train()
        self.n = n  # Layer indices (Sequence) or n last layers (int) to take
        self.autocast_ctx = autocast_ctx
        self.reshape = reshape
        self.return_class_token = return_class_token

    def forward(self, images):
        if not self.tuning:
            with torch.inference_mode():
                with self.autocast_ctx():
                    features = self.feature_model.get_intermediate_layers(
                        images,
                        n=self.n,
                        reshape=self.reshape,
                        return_class_token=self.return_class_token,
                    )
        else:
            with self.autocast_ctx():
                features = self.feature_model.get_intermediate_layers(
                    images,
                    n=self.n,
                    reshape=self.reshape,
                    return_class_token=self.return_class_token,
                )
        return features


class ProgressiveUpHead(nn.Module):
    """
        Features:
        1. Compresses multi-layer features first.
        2. Progressively upsamples from H/16 to H/1.
        3. Uses Conv blocks to refine details at each scale.
        """

    def __init__(
            self,
            in_channels,
            n_output_channels,
            hidden_dim=256,
            use_layernorm=True,
            use_cls_token=False,
            dropout=0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.total_channels = sum(in_channels)
        if use_cls_token:
            self.total_channels *= 2

        self.n_output_channels = n_output_channels
        self.use_cls_token = use_cls_token

        if use_layernorm:
            self.norm = nn.GroupNorm(1, self.total_channels)
        else:
            self.norm = nn.Identity()

        self.dropout = nn.Dropout3d(dropout)

        self.fusion_conv = nn.Sequential(
            nn.Conv3d(self.total_channels, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.up1 = UpBlock3D(hidden_dim, hidden_dim // 2)
        self.up2 = UpBlock3D(hidden_dim // 2, hidden_dim // 4)
        self.up3 = UpBlock3D(hidden_dim // 4, hidden_dim // 8)
        self.up4 = UpBlock3D(hidden_dim // 8, 32)
        self.classifier = nn.Conv3d(32, n_output_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if hasattr(self.classifier, "bias") and self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

    def _transform_inputs(self, inputs):
        """
        Align all inputs to the resolution of the deepest layer.
        """
        # But we align them just in case.
        target_size = inputs[-1].shape[2:]

        resized_inputs = []
        for x in inputs:
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="trilinear", align_corners=False)
            resized_inputs.append(x)

        return torch.cat(resized_inputs, dim=1)

    def _forward_feature(self, inputs):
        # Handle CLS token if present (Same logic as yours)
        inputs = list(inputs)
        out = []
        for x in inputs:
            if self.use_cls_token:
                patch, cls_token = x
                cls_token = cls_token[:, :, None, None, None].expand_as(patch)
                x = torch.cat([patch, cls_token], dim=1)
            else:
                if x.ndim == 2:
                    x = x[:, :, None, None, None]
            out.append(x)

        # Concatenate all layers
        x = self._transform_inputs(out)
        return x

    def forward(self, inputs):
        # inputs[0] shape: [B, C, D, H, W]
        D = inputs[0].shape[2]
        x = self._forward_feature(inputs)

        x = self.dropout(x)
        x = self.norm(x)

        x = self.fusion_conv(x)

        x = self.up1(x)  # -> H/8
        x = self.up2(x)  # -> H/4
        x = self.up3(x)  # -> H/2
        x = self.up4(x)  # -> H/1

        logits = self.classifier(x)

        return logits


class UpBlock3D(nn.Module):
    """
    Upsample only H,W. Keep D unchanged.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


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
