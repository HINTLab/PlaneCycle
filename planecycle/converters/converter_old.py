
class DinoVisionTransformerConverter(DinoVisionTransformer):
    """Convert a 2D backbone to a 3D PlaneCycle model.

    """

    _SUPPORTED = ('DINOv3',)

    def __init__(
        self,
        cycle_order: Tuple[str, ...] = ('HW', 'DH', 'DW'),
        pool_method: Literal["PCg", "PCm"] = "PCg",
    ):
        self.cycle_order   = cycle_order
        self.pool_method   = pool_method
        self.g_len  = self.n_storage_tokens + 1


    def forward(self, x: Tensor):
        """
        Args:
            x: (B, C, D, H, W)
        """
        B, _, D, _, _ = x.shape
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)  # (B,C,D,H,W) → (BD,C,H,W)
        x, hw_tuple = self.prepare_tokens_with_masks(x)
        B_D, H, W, C = x.shape

        xf = x[:, self.g_len:, :].reshape(B, D, H, W, C)
        xg = x[:, :self.g_len, :].reshape(B, D, self.g_len, C)

        for blk_id, blk in  enumerate(self.blocks):
            plane = self.cycle_order[blk_id % len(self.cycle_order)]
            rope_sincos = self.get_rope(plane, D, H, W)

            xf, xg = self.planecycleop(x = xf, g = xg, plane=plane,
                f_layer = lambda t: self.blk2d(t, rope_sincos))

        return self.process_output(xf, xg)

    def get_rope(self, plane, D, H, W):
        rope_H, rope_W = {
            "HW": (H, W),
            "DW": (D, W),
            "DH": (D, H),
        }[plane]

        if self.rope_embed is not None:
            rope_sincos = self.rope_embed(H=rope_H, W=rope_W)
        else:
            rope_sincos = None

        return rope_sincos

    def process_output(self, xf, xg):
        """Process the output of the backbone.  """
        x_norm_patch = self.norm(xf)
        x_norm_cls = self.norm(xg[:, 0])

        return x_norm_cls, x_norm_patch