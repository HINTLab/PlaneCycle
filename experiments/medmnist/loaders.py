"""Model assembly: build backbone + head per args and load pretrained weights."""

import os

import torch

from experiments.baselines.dinov3.convnext.convnext import BaselineConvNeXt
from experiments.baselines.dinov3.vit.vision_transformer import BaselineViT
from dinov3.models.convnext import ConvNeXt
from dinov3.models.vision_transformer import DinoVisionTransformer
from planecycle.converter import planecycle_converter

from classifiers import CTFMClassifier, Dinov3Classifier, SpectreClassifier
from config import MODEL_WEIGHTS_MAP, arch_kwargs


def _apply_training_method(model, args, device):
    """Shared tail of every loader: freeze the backbone for linear probing,
    report what stays trainable, move to device."""
    print(model)
    if args.training_method == "LP":
        for param in model.backbone.parameters():
            param.requires_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"[*requires_grad*] {name}: {param.shape}")
    return model.to(device)


def load_model(args, device, n_classes):
    """Build a DINOv3-family backbone (ViT or ConvNeXt) with the requested
    method and load pretrained weights.

    Baseline{ViT,ConvNeXt} handle all block types through one flag; their
    state-dict keys equal the plain backbone's in every mode, so the official
    checkpoints load with strict=True. TriSlice (2.5D) uses a plain Slice2D
    backbone — the neighbouring-slice stacking happens in Dinov3Classifier.
    """
    backbone_block_type = "Slice2D" if args.block_type == "TriSlice" else args.block_type
    # TriSlice is by definition Slice2D + neighbouring-slice channels
    channels_data = "neighbors" if args.block_type == "TriSlice" else args.channels_data
    cycle_order = tuple(args.cycle_order) if args.cycle_order else ("HW", "DW", "DH", "HW")
    is_convnext = "convnext" in args.arch

    weights_path = os.path.join(args.weight_dir, MODEL_WEIGHTS_MAP[args.arch])
    print(f"[*] Loading weights: {weights_path}")
    state_dict = torch.load(weights_path, map_location="cpu")

    if args.block_type == "ACS":
        # ACS convolution baseline (ConvNeXt only): a native-3D ConvNeXt whose
        # Conv2d are ACSConv. Same param shapes as the plain ConvNeXt, so the
        # 2D pretrained weights load with strict=True; output is (xf, xcls).
        from experiments.baselines.dinov3.convnext.convnext_acs import ConvNeXt as ACSConvNeXt

        backbone = ACSConvNeXt(**arch_kwargs(args.arch))
        backbone.load_state_dict(state_dict, strict=True)
        embed_dim = backbone.embed_dim
    elif args.block_type == "PlaneCycle" and not args.disable_converter:
        # primary path: the released planecycle_converter over a plain backbone
        plain_cls = ConvNeXt if is_convnext else DinoVisionTransformer
        backbone = plain_cls(**arch_kwargs(args.arch))
        backbone.load_state_dict(state_dict, strict=True)
        embed_dim = backbone.embed_dim
        backbone = planecycle_converter(
            backbone, cycle_order=cycle_order, pool_method=args.pool_method or "PCg"
        )
    else:
        # Slice2D / Flatten3D / TriSlice — and PlaneCycle with
        # --disable_converter (BaselineViT's converter-equivalent mode)
        if is_convnext:
            backbone = BaselineConvNeXt(
                block_type=backbone_block_type,
                cycle_order=cycle_order,
                **arch_kwargs(args.arch),
            )
        else:
            backbone = BaselineViT(
                block_type=backbone_block_type,
                cycle_order=cycle_order,
                pool_method=args.pool_method or "PCg",
                **arch_kwargs(args.arch),
            )
        backbone.load_state_dict(state_dict, strict=True)
        embed_dim = backbone.embed_dim

    model = Dinov3Classifier(
        backbone=backbone,
        n_classes=n_classes,
        embed_dim=embed_dim,
        final_pool_method=args.final_pool_method,
        block_type=args.block_type,
        final_slices=args.D_slices,
        upsample_scale=args.upsample_scale,
        channels_data=channels_data,
    )
    return _apply_training_method(model, args, device)


def load_spectre_model(args, device, n_classes):
    # local import: SPECTRE needs extra packages the base environment omits
    from experiments.baselines.spectre.models.vision_transformer import vit_large_patch16_128

    print(f"[*] Loading SPECTRE backbone from: {args.spectre_weight_path}")
    backbone = vit_large_patch16_128(
        checkpoint_path_or_url=args.spectre_weight_path,
        num_classes=0,
        global_pool="",
        pos_embed="rope",
        rope_kwargs={"base": 1000.0},
        init_values=1.0,
    )
    embed_dim = backbone.embed_dim  # 1080

    model = SpectreClassifier(
        backbone=backbone, n_classes=n_classes, embed_dim=embed_dim,
    )
    return _apply_training_method(model, args, device)


def load_ctfm_model(args, device, n_classes):
    from lighter_zoo import SegResEncoder

    backbone = SegResEncoder.from_pretrained("project-lighter/ct_fm_feature_extractor")
    embed_dim = backbone.init_filters * (2 ** (len(backbone.blocks_down) - 1))
    model = CTFMClassifier(
        backbone=backbone, n_classes=n_classes, embed_dim=embed_dim,
    )
    return _apply_training_method(model, args, device)


