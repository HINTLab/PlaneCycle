import argparse
import os
import random
import time
from collections import OrderedDict
from copy import deepcopy

import medmnist
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import wandb
from PIL import Image
from medmnist import INFO
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision.transforms import v2
from tqdm import tqdm, trange

from planecycle.converters.converter import PlaneCycleConverter
from planecycle.converters.random_plane_converter import RandomPlaneCycleConverter
from planecycle.converters.mlp_plane_converter import MLPPlaneCycleConverter

from config import MODEL_WEIGHTS_MAP


def set_rng_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(gpu_ids: str) -> torch.device:
    visible_gpu_ids = [int(gid) for gid in gpu_ids.split(",") if int(gid) >= 0]
    if visible_gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_gpu_ids[0])
    device = (
        torch.device(f"cuda:{visible_gpu_ids[0]}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"[*] Running on device: {device}")
    return device


class Transform3D(nn.Module):
    def __init__(self, mode="val", resolution=64, target_resolution=64):
        super().__init__()
        self.mode = "train" if "train" in mode else "val"
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        )

        need_resize = target_resolution != resolution
        pad_size = max(1, 4 * (target_resolution // resolution)) if need_resize else 4

        core_spatial_aug = [
            v2.RandomCrop(
                size=target_resolution, padding=pad_size, padding_mode="reflect"
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomChoice(
                [
                    v2.RandomRotation(degrees=(90, 90)),
                    v2.RandomRotation(degrees=(180, 180)),
                    v2.RandomRotation(degrees=(270, 270)),
                    v2.Identity(),
                ]
            ),
        ]

        if need_resize:
            self.train_aug = v2.Compose(
                [
                    v2.Resize(
                        size=target_resolution,
                        interpolation=v2.InterpolationMode.BILINEAR,
                    ),
                    *core_spatial_aug,
                ]
            )
            self.val_test_aug = v2.Resize(
                size=target_resolution, interpolation=v2.InterpolationMode.BILINEAR
            )
        else:
            self.train_aug = v2.Compose(core_spatial_aug)
            self.val_test_aug = v2.Identity()

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = np.array(x) if isinstance(x, Image.Image) else x
            x = torch.from_numpy(x).float()
        if x.max() > 100:
            x = x / 255
        if x.ndim == 3:
            x = x.unsqueeze(0)
        elif x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        x = self.train_aug(x) if self.mode == "train" else self.val_test_aug(x)
        if x.shape[0] == 1:
            x = x.expand(3, -1, -1, -1)
        return (x - self.mean) / self.std


class Transform3DSPECTRE(nn.Module):
    """Same spatial augmentation as Transform3D but for SPECTRE:
    - Single channel (no RGB expansion)
    - No ImageNet normalization — SPECTRE expects values in [0, 1]
    """

    def __init__(self, mode="val", resolution=64, target_resolution=64):
        super().__init__()
        self.mode = "train" if "train" in mode else "val"

        need_resize = target_resolution != resolution
        pad_size = max(1, 4 * (target_resolution // resolution)) if need_resize else 4

        core_spatial_aug = [
            v2.RandomCrop(
                size=target_resolution, padding=pad_size, padding_mode="reflect"
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomChoice(
                [
                    v2.RandomRotation(degrees=(90, 90)),
                    v2.RandomRotation(degrees=(180, 180)),
                    v2.RandomRotation(degrees=(270, 270)),
                    v2.Identity(),
                ]
            ),
        ]

        if need_resize:
            self.train_aug = v2.Compose(
                [
                    v2.Resize(
                        size=target_resolution,
                        interpolation=v2.InterpolationMode.BILINEAR,
                    ),
                    *core_spatial_aug,
                ]
            )
            self.val_test_aug = v2.Resize(
                size=target_resolution, interpolation=v2.InterpolationMode.BILINEAR
            )
        else:
            self.train_aug = v2.Compose(core_spatial_aug)
            self.val_test_aug = v2.Identity()

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = np.array(x) if isinstance(x, Image.Image) else x
            x = torch.from_numpy(x).float()
        if x.max() > 100:
            x = x / 255.0
        if x.ndim == 3:
            x = x.unsqueeze(0)
        elif x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        x = self.train_aug(x) if self.mode == "train" else self.val_test_aug(x)
        # Single channel, values in [0, 1] — no ImageNet normalize
        return x


def upsample_hw(x, scale_factor=2, mode="trilinear"):
    """
    Upsample only H and W dimensions, keep D unchanged.

    Args:
        x: (B, C, D, H, W)
        scale_factor: int or float
        mode: "trilinear" or "nearest"

    Returns:
        x: (B, C, D, H*scale, W*scale)
    """
    x = F.interpolate(
        x,
        scale_factor=(1, scale_factor, scale_factor),
        mode=mode,
        align_corners=False if mode in ["trilinear", "bilinear"] else None,
    )
    return x


class TriPlaneHead(nn.Module):
    """Three-plane 3D feature extraction via independent Slice2D forwards.

    For each orthogonal plane, folds the slice axis into the batch dimension,
    runs the full 2D backbone, mean-pools CLS tokens over slices → (B, C).
    The three (B, C) vectors are summed, then classified.

    Planes:
        HW: fold D → (B*D, 3, H, W) → backbone → mean CLS over D → (B, C)
        DW: fold H → (B*H, 3, D, W) → backbone → mean CLS over H → (B, C)
        DH: fold W → (B*W, 3, D, H) → backbone → mean CLS over W → (B, C)

    use_mlp=True  (TriPlane):    sum → Linear(C, C) → GELU → Linear(C, n_classes)
    use_mlp=False (TriPlaneSum): sum → Linear(C, n_classes)
    Backbone weights are shared across all three planes (training-free).
    """

    def __init__(
        self,
        *,
        backbone,
        n_classes,
        embed_dim,
        upsample_scale=1,
        use_mlp=True,
        use_cat=False,
        independent_backbones=False,
    ):
        import copy

        super().__init__()
        self.upsample_scale = upsample_scale
        self.use_cat = use_cat
        self.independent_backbones = independent_backbones
        if independent_backbones:
            print(f"[*] TriPlaneHead: using independent backbones")
            self.backbone_hw = backbone
            self.backbone_dw = copy.deepcopy(backbone)
            self.backbone_dh = copy.deepcopy(backbone)
        else:
            self.backbone = backbone
        in_dim = embed_dim * 3 if use_cat else embed_dim
        if use_mlp:
            self.head = nn.Sequential(
                nn.Linear(in_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, n_classes),
            )
        else:
            self.head = nn.Linear(in_dim, n_classes)
            print(f"[*] {'TriPlaneCat' if use_cat else 'TriPlaneSum'}: no MLP head")

    def _forward_plane(
        self, x: torch.Tensor, plane: str, backbone=None
    ) -> torch.Tensor:
        """Run backbone on one plane; returns mean CLS (B, C).

        Args:
            x:        (B, 3, D, H, W)
            plane:    'HW' | 'DW' | 'DH'
            backbone: backbone module to use; defaults to self.backbone
        Returns:
            (B, C) — mean CLS token over the slice axis of this plane
        """
        if backbone is None:
            backbone = self.backbone
        B = x.shape[0]
        if plane == "HW":
            x_in, n_slices = x, x.shape[2]  # fold D
        elif plane == "DW":
            x_in = x.permute(0, 1, 3, 2, 4).contiguous()  # (B,3,H,D,W) — fold H
            n_slices = x.shape[3]
        else:  # DH
            x_in = x.permute(0, 1, 4, 2, 3).contiguous()  # (B,3,W,D,H) — fold W
            n_slices = x.shape[4]
        # backbone returns (patch_tokens, cls_tokens)
        # cls_tokens: (B*n_slices, 1, C)
        _, cls = backbone(x_in)
        return cls.squeeze(1).reshape(B, n_slices, -1).mean(dim=1)  # (B, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample_scale > 1:
            x = upsample_hw(x, scale_factor=self.upsample_scale)
        if self.independent_backbones:
            hw = self._forward_plane(x, "HW", self.backbone_hw)
            dw = self._forward_plane(x, "DW", self.backbone_dw)
            dh = self._forward_plane(x, "DH", self.backbone_dh)
        else:
            hw = self._forward_plane(x, "HW")
            dw = self._forward_plane(x, "DW")
            dh = self._forward_plane(x, "DH")
        if self.use_cat:
            feat = torch.cat([hw, dw, dh], dim=-1)  # (B, 3C)
        else:
            feat = hw + dw + dh  # (B, C)
        return self.head(feat)


class Dinov3Linear(nn.Module):
    def __init__(
        self,
        *,
        backbone,
        n_classes,
        embed_dim,
        final_pool_method: str = "learn_to_pool",
        upsample_scale=False,
        block_type="PlaneCycle",
        final_slices=64,
        channels_data: str = "repeated",
    ):
        super().__init__()
        self.backbone = backbone
        self.final_pool_method = final_pool_method
        if self.final_pool_method == "learn_to_pool":
            self.pool_slices = nn.Linear(final_slices, 1)
        self.linear_head = nn.Linear(embed_dim, n_classes)
        self.upsample_scale = upsample_scale
        self.block_type = block_type
        self.channels_data = channels_data

    @staticmethod
    def _make_tri_slice(x: torch.Tensor) -> torch.Tensor:
        """Replace 3 identical grayscale channels with (prev, curr, next) slices.

        Args:
            x: (B, 3, D, H, W) — channels are replicated grayscale
        Returns:
            (B, 3, D, H, W) — channels are slice d-1, d, d+1 (replicate-padded)
        """
        # print(f"[*] TriSlice: _make_tri_slice")
        gray = x[:, 0:1, :, :, :]  # (B, 1, D, H, W)
        # replicate-pad D by 1 on each side → (B, 1, D+2, H, W)
        gray_pad = F.pad(gray, (0, 0, 0, 0, 1, 1), mode="replicate")
        D = x.shape[2]
        prev = gray_pad[:, :, 0:D, :, :]
        curr = gray_pad[:, :, 1 : D + 1, :, :]
        next_ = gray_pad[:, :, 2 : D + 2, :, :]
        return torch.cat([prev, curr, next_], dim=1)  # (B, 3, D, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x: (B, 3, D, H, W)
        Returns:
            xf: spatial features (B, D, H, W, C)
            xcls: CLS token per slice (B, D, C)
        """
        B, _, D, _, _ = x.shape
        if self.upsample_scale > 1:
            x = upsample_hw(x, scale_factor=self.upsample_scale)  # → (B, C, D, 2H, 2W)
        if self.block_type == "TriSlice" or self.channels_data == "neighbors":
            x = self._make_tri_slice(x)
        xf, xcls = self.backbone(x)

        if self.block_type == "Flatten3D":
            return self.linear_head(xcls)

        if self.final_pool_method == "learn_to_pool":
            xcls = self.pool_slices(xcls.permute(0, 2, 1)).squeeze(
                -1
            )  # (B, D, C) -> (B, C)
        elif self.final_pool_method == "mean":
            xcls = xcls.mean(dim=1)  # (B, D, C) -> (B, C)
        else:
            raise ValueError(f"Unknown final_pool_method: {self.final_pool_method}")

        return self.linear_head(xcls)


class SpectreLinearHead(nn.Module):
    """Linear probe head for SPECTRE backbone.

    Wraps a SPECTRE VisionTransformer (global_pool='') and applies a linear
    classifier on the CLS token. Handles the axis permutation from our
    (B, C, D, H, W) convention to SPECTRE's (B, C, H, W, D).
    """

    def __init__(self, *, backbone, n_classes, embed_dim):
        super().__init__()
        self.backbone = backbone
        self.linear_head = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, D, H, W) → (B, 1, H, W, D) for SPECTRE
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        features = self.backbone(x)  # (B, T+1, embed_dim)  global_pool=''
        cls = features[:, 0, :]  # CLS token → (B, embed_dim)
        return self.linear_head(cls)


class CTFMLinearHead(nn.Module):
    """Linear probe head for CT-FM (SegResEncoder) backbone.

    Applies global average pooling on the deepest encoder feature map
    to produce a fixed-length embedding, then classifies with a linear head.
    Input: (B, 1, D, H, W), values in [0, 1].
    """

    def __init__(self, *, backbone, n_classes, embed_dim):
        super().__init__()
        self.backbone = backbone
        self.linear_head = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, D, H, W)
        features = self.backbone(x)[-1]  # (B, C, d, h, w)
        cls = F.adaptive_avg_pool3d(features, 1).flatten(1)  # (B, C)
        return self.linear_head(cls)


def load_model(args, device, n_classes):
    # TriPlane / TriPlaneSum / TriPlaneCat / TriSlice use a plain Slice2D backbone
    _backbone_block_type = (
        "Slice2D"
        if args.block_type in ("TriPlane", "TriPlaneSum", "TriPlaneCat", "TriSlice")
        else args.block_type
    )
    backbone = torch.hub.load(
        args.repo_path,
        args.arch,
        source="local",
        pretrained=False,
        block_type=_backbone_block_type,
        disable_converter=args.disable_converter,
        pool_D=args.pool_D,
    )

    weights_path = os.path.join(args.weight_dir, MODEL_WEIGHTS_MAP[args.arch])
    print(f"[*] Loading weights: {weights_path}")

    pretrained_weights = torch.load(weights_path, map_location=device)
    if args.block_type == "Flatten3D":
        unwanted_key = "rope_embed.periods"
        if unwanted_key in pretrained_weights:
            print(
                f"Removing {unwanted_key} from state_dict to avoid dimension mismatch."
            )
            del pretrained_weights[unwanted_key]
    if args.block_type != "Conv3D":
        backbone.load_state_dict(pretrained_weights, strict=True)

    embed_dim = backbone.embed_dim
    if args.block_type in ("TriPlane", "TriPlaneSum", "TriPlaneCat"):
        model = TriPlaneHead(
            backbone=backbone,
            n_classes=n_classes,
            embed_dim=embed_dim,
            upsample_scale=args.upsample_scale,
            use_mlp=(args.block_type == "TriPlane"),
            use_cat=(args.block_type == "TriPlaneCat"),
            independent_backbones=args.independent_backbones,
        )
    else:
        if args.block_type == "PlaneCycle":
            if args.pool_method == "PCmlp":
                backbone = MLPPlaneCycleConverter(
                    backbone=backbone,
                    cycle_order=args.cycle_order,
                    pool_method="PCg",
                    resolution=args.target_resolution,
                    patch_size=backbone.patch_size,
                )
            else:
                converter_cls = (
                    RandomPlaneCycleConverter
                    if args.random_plane
                    else PlaneCycleConverter
                )
                backbone = converter_cls(
                    backbone=backbone,
                    cycle_order=args.cycle_order,
                    pool_method=args.pool_method,
                )

        model = Dinov3Linear(
            backbone=backbone,
            n_classes=n_classes,
            embed_dim=embed_dim,
            final_pool_method=args.final_pool_method,
            block_type=args.block_type,
            final_slices=args.D_slices,
            upsample_scale=args.upsample_scale,
            channels_data=args.channels_data,
        )
    print(model)

    if args.training_method == "LP":
        if isinstance(model, TriPlaneHead) and model.independent_backbones:
            for bb in (model.backbone_hw, model.backbone_dw, model.backbone_dh):
                for param in bb.parameters():
                    param.requires_grad = False
        else:
            for name, param in model.backbone.named_parameters():
                param.requires_grad = False
        # slice_proj is a new 3D-adaptation parameter, not pretrained — keep trainable
        if isinstance(model.backbone, MLPPlaneCycleConverter):
            for blk in model.backbone.backbone.blocks:
                if hasattr(blk, "slice_proj"):
                    blk.slice_proj.weight.requires_grad = True
                    blk.slice_proj.bias.requires_grad = True

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"[*requires_grad*] {name}: {param.shape}")

    model.to(device)
    return model


def load_spectre_model(args, device, n_classes):
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "models"))
    from spectre.models.vision_transformer import vit_large_patch16_128

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

    model = SpectreLinearHead(
        backbone=backbone, n_classes=n_classes, embed_dim=embed_dim
    )
    print(model)

    if args.training_method == "LP":
        for param in model.backbone.parameters():
            param.requires_grad = False

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"[*requires_grad*] {name}: {param.shape}")

    model.to(device)
    return model


def load_ctfm_model(args, device, n_classes):
    from lighter_zoo import SegResEncoder

    backbone = SegResEncoder.from_pretrained("project-lighter/ct_fm_feature_extractor")
    embed_dim = backbone.init_filters * (2 ** (len(backbone.blocks_down) - 1))
    model = CTFMLinearHead(backbone=backbone, n_classes=n_classes, embed_dim=embed_dim)
    print(model)

    if args.training_method == "LP":
        for param in model.backbone.parameters():
            param.requires_grad = False

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"[*requires_grad*] {name}: {param.shape}")

    model.to(device)
    return model


def build_scheduler(args, optimizer):
    if args.scheduler == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs, eta_min=args.min_lr
        )
    if args.scheduler == "WarmupCosineAnnealingLR":
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.001,
            end_factor=1.0,
            total_iters=args.warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=(args.num_epochs - args.warmup_epochs), eta_min=args.min_lr
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[args.warmup_epochs],
        )
    milestones = [int(0.9 * args.num_epochs), int(0.95 * args.num_epochs)]
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1
    )


def init_wandb_run(args):
    return wandb.init(
        entity=args.entity,
        project=args.project_name,
        name=args.run_name
        or f"{args.data_flag}_{args.arch}_{args.block_type}_{args.pool_method}",
        config={
            "dataset": args.data_flag,
            "architecture": args.arch,
            "block_type": args.block_type,
            "pool_method": args.pool_method,
            "final_pool_method": args.final_pool_method,
            "training_method": args.training_method,
            "cycle_order": args.cycle_order,
            "epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.max_lr,
            "min_lr": args.min_lr,
            "weight_decay": args.weight_decay,
            "scheduler": args.scheduler,
            "warmup_epochs": args.warmup_epochs,
            "target_resolution": args.target_resolution,
            "size": args.size,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "upsample_scale": args.upsample_scale,
            "disable_converter": args.disable_converter,
            "D_slices": args.D_slices,
            "as_rgb": args.as_rgb,
            "pool_D": args.pool_D,
            "model_family": getattr(args, "model_family", "dinov3"),
            "spectre_weight_path": getattr(args, "spectre_weight_path", None),
        },
    )


def train(model, train_loader, task, criterion, optimizer, device):
    total_loss = []
    model.train()
    for inputs, targets in tqdm(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        targets = (
            targets.to(torch.float32).to(device)
            if task == "multi-label, binary-class"
            else torch.squeeze(targets, 1).long().to(device)
        )
        loss = criterion(outputs, targets)
        total_loss.append(loss.item())

        loss.backward()
        optimizer.step()

    return sum(total_loss) / len(total_loss)


def test(model, evaluator, data_loader, task, criterion, device, run, save_folder=None):
    model.eval()
    total_loss, y_score = [], torch.tensor([]).to(device)

    with torch.no_grad():
        for inputs, targets in tqdm(data_loader):
            outputs = model(inputs.to(device))
            if task == "multi-label, binary-class":
                targets = targets.to(torch.float32).to(device)
                loss = criterion(outputs, targets)
                outputs = nn.Sigmoid()(outputs).to(device)
            else:
                targets = torch.squeeze(targets, 1).long().to(device)
                loss = criterion(outputs, targets)
                outputs = nn.Softmax(dim=1)(outputs).to(device)
            total_loss.append(loss.item())
            y_score = torch.cat((y_score, outputs), 0)

    y_score = y_score.detach().cpu().numpy()
    auc, acc = evaluator.evaluate(y_score, save_folder, run)
    test_loss = sum(total_loss) / len(total_loss)
    return [test_loss, auc, acc]


def main(args):
    set_rng_seed(args.seed)
    device = get_device(args.gpu_ids)
    output_root = os.path.join(
        args.output_root, args.data_flag, time.strftime("%y%m%d_%H%M%S")
    )
    os.makedirs(output_root, exist_ok=True)

    # Prepare data
    print("==> Preparing data...")
    info = INFO[args.data_flag]
    task, n_classes = info["task"], len(info["label"])
    data_class = getattr(medmnist, info["python_class"])

    if args.model_family in ["spectre", "ctfm"]:
        train_transform = Transform3DSPECTRE(
            mode="train", resolution=args.size, target_resolution=args.target_resolution
        )
        eval_transform = Transform3DSPECTRE(
            mode="val", resolution=args.size, target_resolution=args.target_resolution
        )
        dataset_kwargs = dict(download=args.download, as_rgb=False, size=args.size)
    else:
        train_transform = Transform3D(
            mode="train", resolution=args.size, target_resolution=args.target_resolution
        )
        eval_transform = Transform3D(
            mode="val", resolution=args.size, target_resolution=args.target_resolution
        )
        dataset_kwargs = dict(
            download=args.download, as_rgb=args.as_rgb, size=args.size
        )
    train_dataset = data_class(
        split="train", transform=train_transform, **dataset_kwargs
    )
    train_dataset_at_eval = data_class(
        split="train", transform=eval_transform, **dataset_kwargs
    )
    val_dataset = data_class(split="val", transform=eval_transform, **dataset_kwargs)
    test_dataset = data_class(split="test", transform=eval_transform, **dataset_kwargs)

    def _worker_init_fn(worker_id):
        worker_seed = args.seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    _g = torch.Generator()
    _g.manual_seed(args.seed)
    dl_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        worker_init_fn=_worker_init_fn,
    )
    train_loader = data.DataLoader(
        train_dataset, shuffle=True, generator=_g, **dl_kwargs
    )
    train_loader_at_eval = data.DataLoader(
        train_dataset_at_eval, shuffle=False, **dl_kwargs
    )
    val_loader = data.DataLoader(val_dataset, shuffle=False, **dl_kwargs)
    test_loader = data.DataLoader(test_dataset, shuffle=False, **dl_kwargs)

    train_evaluator = medmnist.Evaluator(args.data_flag, "train", size=args.size)
    val_evaluator = medmnist.Evaluator(args.data_flag, "val", size=args.size)
    test_evaluator = medmnist.Evaluator(args.data_flag, "test", size=args.size)

    criterion = (
        nn.BCEWithLogitsLoss()
        if task == "multi-label, binary-class"
        else nn.CrossEntropyLoss()
    )

    print("==> Building and training model...")
    if args.model_family == "spectre":
        model = load_spectre_model(args, device, n_classes)
    elif args.model_family == "ctfm":
        model = load_ctfm_model(args, device, n_classes)
    else:
        model = load_model(args, device, n_classes).to(device)

    if args.model_path is not None:
        model.load_state_dict(
            torch.load(args.model_path, map_location=device)["net"], strict=True
        )
        train_metrics = test(
            model,
            train_evaluator,
            train_loader_at_eval,
            task,
            criterion,
            device,
            args.run,
            output_root,
        )
        val_metrics = test(
            model,
            val_evaluator,
            val_loader,
            task,
            criterion,
            device,
            args.run,
            output_root,
        )
        test_metrics = test(
            model,
            test_evaluator,
            test_loader,
            task,
            criterion,
            device,
            args.run,
            output_root,
        )
        print(
            f"train  auc: {train_metrics[1]:.5f}  acc: {train_metrics[2]:.5f}\nval    auc: {val_metrics[1]:.5f}  acc: {val_metrics[2]:.5f}\ntest   auc: {test_metrics[1]:.5f}  acc: {test_metrics[2]:.5f}"
        )

    if args.num_epochs == 0:
        return

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.max_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(args, optimizer)

    logs = ["loss", "auc", "acc"]
    train_logs = ["train_" + log for log in logs]
    val_logs = ["val_" + log for log in logs]
    test_logs = ["test_" + log for log in logs]
    log_dict = OrderedDict.fromkeys(train_logs + val_logs + test_logs, 0)

    best_auc, best_epoch, best_model = 0.0, 0, deepcopy(model)

    for epoch in trange(args.num_epochs):
        train_loss = train(model, train_loader, task, criterion, optimizer, device)
        train_metrics = test(
            model,
            train_evaluator,
            train_loader_at_eval,
            task,
            criterion,
            device,
            args.run,
        )
        val_metrics = test(
            model, val_evaluator, val_loader, task, criterion, device, args.run
        )
        test_metrics = test(
            model, test_evaluator, test_loader, task, criterion, device, args.run
        )

        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        print(f"Learning Rate from scheduler: {lr:.6f}")

        for i, key in enumerate(train_logs):
            log_dict[key] = train_metrics[i]
        for i, key in enumerate(val_logs):
            log_dict[key] = val_metrics[i]
        for i, key in enumerate(test_logs):
            log_dict[key] = test_metrics[i]

        for key, value in log_dict.items():
            print(key, value, epoch)

        metric_names = ["loss", "auc", "acc"]
        payload = {"epoch": epoch, "lr": lr}
        payload.update(
            {f"train/{name}": val for name, val in zip(metric_names, train_metrics)}
        )
        payload.update(
            {f"val/{name}": val for name, val in zip(metric_names, val_metrics)}
        )
        payload.update(
            {f"test/{name}": val for name, val in zip(metric_names, test_metrics)}
        )

        wandb.log(payload)

        cur_auc = val_metrics[1]
        if cur_auc >= best_auc:
            best_epoch, best_auc, best_model = epoch, cur_auc, deepcopy(model)
            print(f"cur_best_auc: {best_auc}, cur_best_epoch: {best_epoch}")
            wandb.run.summary.update({"best_auc": best_auc, "best_epoch": best_epoch})
            # torch.save(model.state_dict())

        torch.cuda.empty_cache()

    train_metrics = test(
        best_model,
        train_evaluator,
        train_loader_at_eval,
        task,
        criterion,
        device,
        args.run,
        output_root,
    )
    val_metrics = test(
        best_model,
        val_evaluator,
        val_loader,
        task,
        criterion,
        device,
        args.run,
        output_root,
    )
    test_metrics = test(
        best_model,
        test_evaluator,
        test_loader,
        task,
        criterion,
        device,
        args.run,
        output_root,
    )

    log = f"{args.data_flag}\ntrain  auc: {train_metrics[1]:.5f}  acc: {train_metrics[2]:.5f}\nval    auc: {val_metrics[1]:.5f}  acc: {val_metrics[2]:.5f}\ntest   auc: {test_metrics[1]:.5f}  acc: {test_metrics[2]:.5f}\n"
    print(log)

    summary = {
        "final_train_auc": train_metrics[1],
        "final_train_acc": train_metrics[2],
        "final_val_auc": val_metrics[1],
        "final_val_acc": val_metrics[2],
        "final_test_auc": test_metrics[1],
        "final_test_acc": test_metrics[2],
    }

    wandb.run.summary.update(summary)

    with open(os.path.join(output_root, f"{args.data_flag}_log.txt"), "a") as f:
        f.write(log)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a single-task MedMNIST 3D model."
    )

    # Experiment arguments
    exp = parser.add_argument_group("experiment")
    exp.add_argument(
        "--project_name",
        default="dinov3",
        type=str,
        help="Weights & Biases project name.",
    )
    exp.add_argument(
        "--entity",
        default="<your_wandb_entity>",
        type=str,
        help="Weights & Biases entity or team name.",
    )
    exp.add_argument(
        "--run",
        default="model1",
        type=str,
        help="Suffix used by MedMNIST evaluator output files.",
    )
    exp.add_argument(
        "--run_name", default=None, type=str, help="Optional explicit W&B run name."
    )
    exp.add_argument(
        "--output_root",
        default="./outputs",
        type=str,
        help="Directory for logs and evaluation outputs.",
    )
    exp.add_argument(
        "--gpu_ids",
        default="0",
        type=str,
        help='Comma-separated GPU ids, e.g. "0" or "0,1".',
    )
    exp.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="Number of DataLoader worker processes.",
    )
    exp.add_argument(
        "--seed", default=42, type=int, help="Random seed for reproducibility."
    )
    exp.add_argument(
        "--download",
        action="store_true",
        help="Download MedMNIST data if not found locally.",
    )

    # Dataset arguments
    dset = parser.add_argument_group("dataset")
    dset.add_argument(
        "--data_flag", default="organmnist3d", type=str, help="MedMNIST dataset name."
    )
    dset.add_argument(
        "--size", default=64, type=int, help="Original dataset image size."
    )
    dset.add_argument(
        "--target_resolution",
        default=64,
        type=int,
        help="Target spatial resolution after preprocessing.",
    )
    dset.add_argument(
        "--upsample_scale",
        default=1,
        type=int,
        help="Upsample image before feeding to backbone.",
    )
    dset.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Mini-batch size for training and evaluation.",
    )
    dset.add_argument(
        "--as_rgb",
        action="store_true",
        help="Repeat single-channel volume to 3 channels.",
    )

    # Optimization arguments
    opt = parser.add_argument_group("optimization")
    opt.add_argument(
        "--num_epochs", default=100, type=int, help="Number of training epochs."
    )
    opt.add_argument(
        "--max_lr", default=1e-3, type=float, help="Initial learning rate."
    )
    opt.add_argument(
        "--min_lr",
        default=1e-6,
        type=float,
        help="Minimum learning rate for cosine schedulers.",
    )
    opt.add_argument(
        "--weight_decay", default=1e-2, type=float, help="Weight decay for AdamW."
    )
    opt.add_argument(
        "--warmup_epochs", default=10, type=int, help="Number of warmup epochs."
    )
    opt.add_argument(
        "--scheduler",
        default="WarmupCosineAnnealingLR",
        type=str,
        help="Learning-rate scheduler: MultiStepLR, CosineAnnealingLR, or WarmupCosineAnnealingLR.",
    )

    # Model arguments
    mdl = parser.add_argument_group("model")
    mdl.add_argument(
        "--training_method",
        default="LP",
        type=str,
        help="Training mode: LP(linear probing) or FT(finetune).",
    )
    mdl.add_argument(
        "--channels_data",
        default="repeated",
        type=str,
        help="Training mode: LP(linear probing) or FT(finetune).",
    )
    mdl.add_argument(
        "--repo_path",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "models"),
        type=str,
        help="Local torch.hub repository path.",
    )
    mdl.add_argument(
        "--weight_dir",
        default=None,
        type=str,
        help="Path to pretrained backbone weights.",
    )
    mdl.add_argument(
        "--arch", default="dinov3_vits16", type=str, help="Backbone architecture name."
    )
    mdl.add_argument(
        "--block_type",
        default="PlaneCycle",
        type=str,
        help='Backbone block type or "PlaneCycle".',
    )
    mdl.add_argument(
        "--pool_method",
        default="",
        type=str,
        help="PlaneCycle pooling method, PCg or PCm.",
    )
    mdl.add_argument(
        "--final_pool_method",
        default="learn_to_pool",
        type=str,
        help="Final pooling method: mean, learn_to_pool",
    )
    mdl.add_argument(
        "--D_slices",
        default=64,
        type=int,
        help="Number of depth slices for final pooling head.",
    )
    mdl.add_argument(
        "--final_slices",
        default=64,
        type=int,
        help="Number of depth slices for final pooling head.",
    )
    mdl.add_argument(
        "--concat_patch_token",
        action="store_true",
        help="Concatenate mean patch token to CLS token.",
    )
    mdl.add_argument(
        "--cycle_order",
        nargs="+",
        choices=["HW", "DW", "DH"],
        default=[],
        help="Plane traversal order for PlaneCycle blocks.",
    )
    mdl.add_argument(
        "--random_plane",
        action="store_true",
        default=False,
        help="Randomly select plane per block during training (PlaneCycle only). Last block fixed to HW.",
    )
    mdl.add_argument(
        "--independent_backbones",
        action="store_true",
        default=False,
        help="TriPlane only: give each plane (HW/DW/DH) its own backbone copy. Useful for FT.",
    )
    mdl.add_argument(
        "--disable_converter",
        action="store_true",
        help="Concatenate mean patch token to CLS token.",
    )
    mdl.add_argument(
        "--pool_D",
        action="store_true",
        help="Concatenate mean patch token to CLS token.",
    )

    # SPECTRE arguments
    spe = parser.add_argument_group("spectre")
    spe.add_argument(
        "--model_family",
        default="dinov3",
        choices=["dinov3", "spectre", "ctfm"],
        help='Model family to use. "spectre" activates SPECTRE backbone and Transform3DSPECTRE.',
    )
    spe.add_argument(
        "--spectre_weight_path",
        default=None,
        type=str,
        help="Local path or URL to the SPECTRE SSL-only backbone checkpoint "
        "(spectre_backbone_vit_large_patch16_128_no_vla.pt). "
        "Required when --model_family=spectre.",
    )

    # Evaluation arguments
    eva = parser.add_argument_group("evaluation")
    eva.add_argument(
        "--model_path",
        default=None,
        type=str,
        help="Optional checkpoint path for evaluation or warm start.",
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    for key, value in vars(args).items():
        print(f"{key:20}: {value}")

    wandb_run = init_wandb_run(args)
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

    main(args)

    wandb_run.finish()
