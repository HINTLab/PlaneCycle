import torch
import torch.nn as nn
import torch.nn.functional as F


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
