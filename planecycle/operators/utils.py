import torch
import torch.nn.functional as F


def adaptive_avg_pool_along_dim(
    x: torch.Tensor, output_size: int, dim: int = 1
) -> torch.Tensor:
    """Adaptive average pool along dimension `dim` to `output_size`.
    Args:
        x: Input tensor.
        output_size: Target size for dimension `dim`.
        dim: Dimension to pool (supports negative indexing).
    """
    dim %= x.ndim
    # 相同size可以直接返回
    if x.size(dim) == output_size:
        return x

    # Edge case: input_size == 1 mathematically means replicate the single slice
    # `output_size` times. CUDA's adaptive_avg_pool1d has a known illegal memory
    # access bug with input_size=1, output_size>1; use expand instead.
    if x.size(dim) == 1:
        shape = list(x.shape)
        shape[dim] = output_size
        return x.expand(shape).contiguous()

    # adaptive pool 需要在最后一维，需要交换位置
    x = torch.moveaxis(x, dim, -1)
    *batch_shape, last_dim = x.shape
    x = x.reshape(-1, 1, last_dim)
    x = F.adaptive_avg_pool1d(x, output_size)
    x = x.reshape(*batch_shape, output_size)
    x = torch.moveaxis(x, -1, dim)
    return x
