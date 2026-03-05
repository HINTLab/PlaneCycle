import torch
import torch.nn.functional as F


def adaptive_avg_pool_along_dim(x: torch.Tensor, output_size: int, dim: int = 1) -> torch.Tensor:
    """Adaptive average pool along dimension `dim` to `output_size`.
    Args:
        x: Input tensor.
        output_size: Target size for dimension `dim`.
        dim: Dimension to pool (supports negative indexing).
    """
    dim %= x.ndim
    if x.size(dim) == output_size:
        return x

    x = torch.moveaxis(x, dim, -1)
    *batch_shape, last_dim = x.shape
    x = x.reshape(-1, 1, last_dim)
    x = F.adaptive_avg_pool1d(x, output_size)
    x = x.reshape(*batch_shape, output_size)
    x = torch.moveaxis(x, -1, dim)
    return x
