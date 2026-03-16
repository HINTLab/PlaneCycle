import torch
import torch.nn as nn
import torch.nn.functional as F


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
