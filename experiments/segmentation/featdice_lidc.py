import os
import math
import numpy as np
import pandas as pd
import argparse

import torch
import torch.nn.functional as F
from monai.metrics import DiceMetric
import traceback

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='small')
    parser.add_argument('--mode', type=str, default='PlaneCycle')   # Select from 'PlanceCycle', 'Slice2D' or 'Flatten3D'
    parser.add_argument('--threshold', type=float, default=0.9)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device: {device}")

    from experiments.segmentation.datasets.lidc_dataset import LIDCSegDataset

    input_size, patch_size = 640, 16
    crop_size, move, upsampling_size = input_size, 5, 640
    lidc_folder = ...   # TODO: lidc root dir

    lidc_tr = LIDCSegDataset(crop_size=crop_size, move=move, data_path=lidc_folder, train=True,
                             upsampling_size=upsampling_size)
    lidc_ts = LIDCSegDataset(crop_size=crop_size, move=move, data_path=lidc_folder, train=False,
                             upsampling_size=upsampling_size, original_mask=False)

    print(f"Train set size: {len(lidc_tr)}, test set size: {len(lidc_ts)}")

    from torch.utils.data import DataLoader

    # Datalaoder
    batch_size, num_workers = 2, 1
    loader_tr = DataLoader(lidc_tr, batch_size=batch_size, shuffle=True,
                           pin_memory=(torch.cuda.is_available()), num_workers=num_workers)
    loader_ts = DataLoader(lidc_ts, batch_size=batch_size, shuffle=False,
                           pin_memory=(torch.cuda.is_available()), num_workers=num_workers)

    # Cfgs dir
    MODEL_REPO_PATH = '*/PlaneCycle/models'     # TODO: hubconf root dir
    CHECKPOINT_DIR = ...     # TODO: checkpoint root dir

    if args.model == 'small':
        CHECKPOINT_FILENAME = 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
        MODEL_ARCH = 'dinov3_vits16'
    elif args.model == 'base':
        CHECKPOINT_FILENAME = 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
        MODEL_ARCH = 'dinov3_vitb16'
    elif args.model == 'large':
        CHECKPOINT_FILENAME = 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
        MODEL_ARCH = 'dinov3_vitl16'

    FULL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)

    import sys

    sys.path.append(os.path.abspath("*/PlaneCycle"))    # TODO: project root dir
    from planecycle.converters.converter import PlaneCycleConverter

    # Load model offline
    model = torch.hub.load(
        repo_or_dir=MODEL_REPO_PATH,
        model=MODEL_ARCH,
        source='local',
        pretrained=False,
        block_type=args.mode,
    )

    # Load pretrained weights
    if not os.path.exists(FULL_CHECKPOINT_PATH):
        raise FileNotFoundError(f"[!ERROR!] Fail to find weights path: {FULL_CHECKPOINT_PATH}")

    print(f"[*INFO*] Loading pretrained weights: {CHECKPOINT_FILENAME}")
    pretrained_weights = torch.load(FULL_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(pretrained_weights, strict=False)

    if args.mode == 'PlaneCycle':
        converter = PlaneCycleConverter(
            cycle_order=('HW', 'DW', 'DH', 'HW'),
            pool_method="PCg"
        )
        model = converter(model)

    model.to(device)
    model.eval()

    dice_metric = DiceMetric(
        include_background=False,
        reduction="mean",
    )

    try:
        scores = {i: [] for i in range(1)}

        with torch.inference_mode():
            for i, (x, y) in enumerate(loader_ts):

                print(f"[*INFO* {i + 1}/{len(loader_ts)}] Inferring...")

                x, y = x.to(device), y.to(device)

                # x: [B, D, C, H, W] -> [B, C, D, H, W]
                x = x.permute(0, 2, 1, 3, 4)

                features = model.get_intermediate_layers(
                    x,
                    n=1,  # Select last n layers to output
                )

                # y: [B, D, K, H, W] -> [B, K, D, H, W]
                y = y.permute(0, 2, 1, 3, 4)

                for j, feature in enumerate(features):
                    h_tok = int(math.sqrt(feature.shape[1]))
                    w_tok = h_tok

                    # =================== PlaneCycle & Slice2D ===================
                    if args.mode == 'PlaneCycle' or args.mode == 'Slice2D':
                        feature = (
                            feature.permute(0, 2, 1)
                            .contiguous()
                            .view(feature.shape[0], feature.shape[-1], h_tok, w_tok)
                        )

                        # Recover to volume: [B, C_feat, D, h, w]
                        feature = (
                            feature.view(y.shape[0], y.shape[2], feature.shape[1], *feature.shape[-2:])
                            .permute(0, 2, 1, 3, 4)
                            .contiguous()
                        )
                    # =================== PlaneCycle & Slice2D ===================

                    # ======================== Flatten 3D ========================
                    if args.mode == 'Flatten3D':
                        feature = (
                            feature.permute(0, 2, 1)  # [B, C, DHW]
                            .contiguous()
                            .view(feature.shape[0], feature.shape[-1], y.shape[2], input_size // patch_size,
                                  input_size // patch_size)
                        )
                    # ======================== Flatten 3D ========================

                    B, C_feat, D, h, w = feature.shape
                    d_center = D // 2

                    for b in range(B):
                        # Center pixel as prototype
                        center_slice = feature[b, :, d_center]  # [C, H, W] # [B, C, H, W]
                        # center_vec = center_slice[:, :, H // 2, W // 2]  # [B,C]
                        center_vec = center_slice[:, h // 2, w // 2]
                        center_vec = center_vec
                        feat_grid = feature[b]

                        norm_ref = F.normalize(center_vec, p=2, dim=0)  # [C]
                        norm_feat = F.normalize(feat_grid, p=2, dim=0)  # [C, D, H, W]

                        sim_map = (norm_feat * norm_ref[:, None, None, None]).sum(dim=0)  # [D, H, W]
                        sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + 1e-8)

                        sim_volume = sim_map.unsqueeze(0).unsqueeze(0)
                        # [1, 1, D, h, w]

                        sim_volume = F.interpolate(
                            sim_volume,
                            size=(D, y.shape[-2], y.shape[-1]),
                            mode="trilinear",
                            align_corners=False
                        )

                        y_b = y[b].unsqueeze(0)  # [1, 1, D, H, W]
                        # sim_volume = sim_map # .unsqueeze(0).unsqueeze(0)

                        mask = y_b[0, 0]  # [D,H,W]
                        nonzero = torch.nonzero(mask)

                        if nonzero.numel() == 0:
                            continue

                        d_min, h_min, w_min = nonzero.min(dim=0).values
                        d_max, h_max, w_max = nonzero.max(dim=0).values

                        margin = 5

                        d_min = max(d_min - margin, 0)
                        h_min = max(h_min - margin, 0)
                        w_min = max(w_min - margin, 0)

                        d_max = min(d_max + margin, mask.shape[0] - 1)
                        h_max = min(h_max + margin, mask.shape[1] - 1)
                        w_max = min(w_max + margin, mask.shape[2] - 1)

                        sim_crop = sim_volume[:, :, d_min:d_max + 1, h_min:h_max + 1, w_min:w_max + 1]
                        y_crop = y_b[:, :, d_min:d_max + 1, h_min:h_max + 1, w_min:w_max + 1]

                        sim_volume = (sim_volume > 0.9).float()
                        dice_metric.reset()
                        dice_metric(sim_volume, y_b)
                        dice = dice_metric.aggregate().item()

                        scores[j].append(dice)

        for k, v in scores.items():
            print(f"Layer {k}'s average score: {sum(v) / len(v)}")

    except torch.cuda.OutOfMemoryError:
        print("[!ERROR!] CUDA out of memory!")
        traceback.print_exc()
    except Exception as e:
        print(f"[!ERROR!] {e}")
        traceback.print_exc()
