MODEL_WEIGHTS_MAP = {
    # ViT Series
    "dinov3_vits16": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "dinov3_vits16plus": "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
    "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "dinov3_vith16plus": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
    "dinov3_vit7b16": "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
    # ConvNeXT Series
    "dinov3_convnext_tiny": "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",
    "dinov3_convnext_small": "dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth",
    "dinov3_convnext_base": "dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth",
    "dinov3_convnext_large": "dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth",
}


# Backbone constructor kwargs per hub arch name. Mirrors the entries in
# dinov3/hub/backbones.py so experiments can build Baseline{ViT,ConvNeXt}
# directly (uniform block_type switching + strict=True weight loading).
VIT_COMMON = dict(
    img_size=224, patch_size=16, in_chans=3,
    pos_embed_rope_base=100, pos_embed_rope_normalize_coords="separate",
    pos_embed_rope_rescale_coords=2, pos_embed_rope_dtype="fp32",
    qkv_bias=True, drop_path_rate=0.0, layerscale_init=1.0e-05,
    norm_layer="layernormbf16", ffn_bias=True, proj_bias=True,
    n_storage_tokens=4, mask_k_bias=True,
)
VIT_ARCHS = {
    "dinov3_vits16":     dict(embed_dim=384,  depth=12, num_heads=6,  ffn_ratio=4,   ffn_layer="mlp",      **VIT_COMMON),
    "dinov3_vits16plus": dict(embed_dim=384,  depth=12, num_heads=6,  ffn_ratio=6,   ffn_layer="swiglu",   **VIT_COMMON),
    "dinov3_vitb16":     dict(embed_dim=768,  depth=12, num_heads=12, ffn_ratio=4,   ffn_layer="mlp",      **VIT_COMMON),
    "dinov3_vitl16":     dict(embed_dim=1024, depth=24, num_heads=16, ffn_ratio=4,   ffn_layer="mlp",      **VIT_COMMON),
    "dinov3_vitl16plus": dict(embed_dim=1024, depth=24, num_heads=16, ffn_ratio=6.0, ffn_layer="swiglu",   **VIT_COMMON),
    "dinov3_vith16plus": dict(embed_dim=1280, depth=32, num_heads=20, ffn_ratio=6.0, ffn_layer="swiglu",   **VIT_COMMON),
    "dinov3_vit7b16":    dict(embed_dim=4096, depth=40, num_heads=32, ffn_ratio=3,   ffn_layer="swiglu64", **VIT_COMMON),
}
# dinov3_vit7b16 differs from the common block: no qkv bias
VIT_ARCHS["dinov3_vit7b16"]["qkv_bias"] = False


def arch_kwargs(arch: str) -> dict:
    """Constructor kwargs for a hub arch name (ViT or ConvNeXt)."""
    if "convnext" in arch:
        from dinov3.models.convnext import convnext_sizes  # single source for sizes
        size = arch.replace("dinov3_convnext_", "")
        return dict(in_chans=3, drop_path_rate=0.0, layer_scale_init_value=1e-6,
                    **convnext_sizes[size])
    return dict(VIT_ARCHS[arch])
