import torch
from torch import nn

from planecycle.operators.planecycle_op import PlaneCycleOp


def _make_inputs():
    B, D, H, W, C = 2, 64, 4, 16, 768
    P_prime, g_len = 4, 5

    x = torch.randn(B, D, H, W, C)
    g = torch.randn(B, P_prime, g_len, C)

    f_layer = nn.Sequential(
        nn.Linear(C, C),
        nn.ReLU(),
        nn.Linear(C, C),
    )
    return x, g, f_layer, (B, D, H, W, C, P_prime, g_len)


def test_planecycleop_basic_forward_pcg():
    x, g, f_layer, (B, D, H, W, C, P_prime, g_len) = _make_inputs()

    op = PlaneCycleOp(pool_method="PCg")
    x_out, g_out = op(x, g, f_layer, plane_dim=1)

    assert x_out.shape == x.shape
    assert g_out.shape == (B, D, g_len, C)


def test_planecycleop_pool_method_pcm():
    x, g, f_layer, (B, D, H, W, C, P_prime, g_len) = _make_inputs()

    op = PlaneCycleOp(pool_method="PCm")
    x_out, g_out = op(x, g, f_layer, plane_dim=1)

    assert x_out.shape == x.shape
    assert g_out.shape == (B, D, g_len, C)


def test_planecycleop_different_plane_dim():
    x, g, f_layer, (B, D, H, W, C, P_prime, g_len) = _make_inputs()
    op = PlaneCycleOp(pool_method="PCg")

    for plane_dim in (1, 2, 3):
        x_out, g_out = op(x, g, f_layer, plane_dim=plane_dim)
        assert x_out.shape == x.shape
        assert g_out.shape == (B, x.shape[plane_dim], g_len, C)


def test_planecycleop_gradient_flow():
    x, g, f_layer, _ = _make_inputs()
    x = x.clone().requires_grad_(True)
    g = g.clone().requires_grad_(True)

    op = PlaneCycleOp(pool_method="PCg")
    x_out, g_out = op(x, g, f_layer, plane_dim=1)

    loss = x_out.sum() + g_out.sum()
    loss.backward()

    assert x.grad is not None
    assert g.grad is not None