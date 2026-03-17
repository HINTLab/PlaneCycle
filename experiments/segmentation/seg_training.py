import os
import random
from functools import partial

import numpy as np
from omegaconf import OmegaConf

import wandb
import torch
from torch.utils.data import DataLoader

from monai.metrics import DiceMetric, MeanIoU

from .datasets.lidc_dataset import LIDCSegDataset
from .datasets.mmwhs_dataset import MMWHS_Dataset
from .decoders import build_segmentation_decoder
from .schedulers import build_scheduler
from .loss import MultiSegmentationLoss

from planecycle.converters.converter import PlaneCycleConverter


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.
    The seed of each worker equals to num_worker * rank + worker_id + user_seed
    cfg:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def train_step(
        segmentation_model: torch.nn.Module,
        batch,
        device,
        scaler,
        optimizer,
        optimizer_gradient_clip,
        scheduler,
        criterion,
        model_dtype=torch.float32,
):
    batch_img, gt = batch
    batch_img = batch_img.to(device)
    gt = gt.to(device)

    B, D = gt.shape[0], gt.shape[1]

    # batch_img = batch_img.reshape(batch_img.shape[0] * batch_img.shape[1], batch_img.shape[2], batch_img.shape[3],
    #                               batch_img.shape[4])
    batch_img = batch_img.permute(0, 2, 1, 3, 4).contiguous()
    gt = gt.permute(0, 2, 1, 3, 4).contiguous()     # [B, C, D, H, W]

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast("cuda", dtype=model_dtype, enabled=True if model_dtype is not None else False):
        pred = segmentation_model(batch_img)

    if gt.shape[-3:] != pred.shape[-3:]:
        pred = torch.nn.functional.interpolate(input=pred, size=gt.shape[-3:], mode="trilinear", align_corners=False)

    loss = criterion(pred, gt)

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            (p for p in segmentation_model.parameters() if p.requires_grad),
            optimizer_gradient_clip
        )
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in segmentation_model.parameters() if p.requires_grad),
            optimizer_gradient_clip
        )
        optimizer.step()

    scheduler.step()

    return loss


def validate(
        segmentation_model: torch.nn.Module,
        val_loader,
        criterion,
        device,
        thresholds=[0.001, 0.01] + list(np.arange(0.05, 0.96, 0.05)),
        cfg=None
):
    segmentation_model.eval()

    pre_eval_results = []
    loss = 0

    prob_volumes = []
    gt_volumes = []

    prob_fg_ls = []
    prob_bg_ls = []
    with torch.no_grad():
        for batch in val_loader:
            batch_img, gt = batch
            batch_img = batch_img.to(device)
            gt = gt.to(device)

            B, D, C, H, W = gt.shape

            # batch_img = batch_img.reshape(batch_img.shape[0] * batch_img.shape[1], batch_img.shape[2],
            #                               batch_img.shape[3],
            #                               batch_img.shape[4])
            batch_img = batch_img.permute(0, 2, 1, 3, 4).contiguous()
            gt = gt.permute(0, 2, 1, 3, 4).contiguous()

            logits = segmentation_model(batch_img)

            if logits.shape[-3:] != gt.shape[-3:]:
                logits = torch.nn.functional.interpolate(
                    logits,
                    size=gt.shape[-3:],
                    mode="trilinear",
                    align_corners=False,
                )

            loss += criterion(logits, gt).item()

            if cfg.dataset == 'lidc':
                prob = torch.sigmoid(logits)
            elif cfg.dataset == 'mmwhs':
                prob = torch.softmax(logits, dim=1)
            else:
                raise KeyError(f"{cfg.dataset} is not supported.")

            # prob = prob.view(B, D, C, H, W).permute(0, 2, 1, 3, 4)  # [B, C, D, H, W]
            # gt = gt.view(B, D, C, H, W).permute(0, 2, 1, 3, 4)  # [B, C, D, H, W]

            prob_volumes.append(prob.cpu())
            gt_volumes.append(gt.cpu())

            prob_fg_ls.append(prob[gt == 1].detach().cpu())
            prob_bg_ls.append(prob[gt == 0].detach().cpu())

    prob_all = torch.cat(prob_volumes, dim=0)  # [N, C, D, H, W]
    gt_all = torch.cat(gt_volumes, dim=0).long()

    wandb.log({
        "prob_true_mean": (prob_all * gt_all).sum(dim=1).mean().item(),
        "prob_fg_mean": torch.cat(prob_fg_ls).mean().item() if len(prob_fg_ls) > 0 else 0.0,
        "prob_bg_mean": torch.cat(prob_bg_ls).mean().item() if len(prob_bg_ls) > 0 else 0.0,
    })

    dice_metric = DiceMetric(
        include_background=False,
        reduction="mean",  # Per-volume mean
    )
    miou_metric = MeanIoU(
        include_background=False,
        reduction="mean",
    )

    if cfg.dataset == 'lidc':
        best_thr = None
        best_dice = -1.0
        best_metrics = None
        dice_curve = []
        for thr in thresholds:
            pred = (prob_all > thr).long()

            dice_metric.reset()
            miou_metric.reset()

            dice_metric(pred, gt_all)
            miou_metric(pred, gt_all)

            dice = dice_metric.aggregate().item()
            miou = miou_metric.aggregate().item()

            dice_curve.append((thr, dice))

            if dice > best_dice:
                best_dice = dice
                best_thr = thr
                best_metrics = {
                    "dice": dice,
                    "mIoU": miou,
                }
    elif cfg.dataset == 'mmwhs':
        pred = torch.argmax(prob_all, dim=1)
        gt_all = gt_all.argmax(dim=1)

        num_classes = prob_all.shape[1]

        pred = torch.nn.functional.one_hot(pred, num_classes=num_classes)
        gt_all = torch.nn.functional.one_hot(gt_all, num_classes=num_classes)

        pred = pred.permute(0, 4, 1, 2, 3).float()
        gt_all = gt_all.permute(0, 4, 1, 2, 3).float()

        dice_metric.reset()
        miou_metric.reset()

        dice_metric(pred, gt_all)
        miou_metric(pred, gt_all)

        dice = dice_metric.aggregate().item()
        miou = miou_metric.aggregate().item()

        best_metrics = {
            "dice": dice,
            "mIoU": miou,
        }
        best_thr = None
        dice_curve = None

    segmentation_model.train()

    metrics = {
        "loss": loss / len(val_loader),
        "dice": best_metrics["dice"],
        "mIoU": best_metrics["mIoU"],
        "best_thr": best_thr,
        "dice_curve": dice_curve,
    }

    return metrics


def main(cfg):
    # Set seed
    set_seed(cfg.seed)
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    init_fn = partial(worker_init_fn, num_workers=cfg.data_loader.num_workers, rank=1, seed=cfg.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*INFO*] Current device: {device}")

    # W&B initialization
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    wandb.init(project="Segmentation Training", entity=...,     # TODO
               name=cfg.wandb_name, reinit=True, dir=...)       # TODO

    # Load datasets
    if cfg.dataset == 'lidc':
        dataset_tr = LIDCSegDataset(**cfg.datasets.train)
        dataset_ts = LIDCSegDataset(**cfg.datasets.val)
        img_sam = dataset_ts[0][0]
    elif cfg.dataset == 'mmwhs':
        dataset_tr = MMWHS_Dataset(**cfg.datasets.train)
        dataset_ts = MMWHS_Dataset(**cfg.datasets.val)
        img_sam = dataset_ts[0][0]
    else:
        raise KeyError(f"Unknown dataset: {cfg.dataset}")

    print(f"Train set size: {len(dataset_tr)}, test set size: {len(dataset_ts)}")
    print(f"Sample image shape: {img_sam.shape}.\n"
          f"Value scale: {img_sam.min()} ~ {img_sam.max()}")
    del img_sam


    # Generate dataloaders
    loader_tr = DataLoader(
        dataset=dataset_tr,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        generator=g,
        worker_init_fn=init_fn,
        **cfg.data_loader
    )
    loader_ts = DataLoader(
        dataset=dataset_ts,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        generator=g,
        worker_init_fn=init_fn,
        **cfg.data_loader
    )

    # Load backbone model
    backbone_model = torch.hub.load(
        **cfg.backbone_model.load_model
    )
    # Load pretrained weights
    checkpoint_path = cfg.backbone_model.checkpoint_path
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"[!ERROR!] Fail to find weights path: {checkpoint_path}")
    pretrained_weights = torch.load(checkpoint_path, map_location=device)
    backbone_model.load_state_dict(pretrained_weights, strict=False)

    if cfg.backbone_model.load_model.block_type == 'PlaneCycle':
        converter = PlaneCycleConverter(
            **cfg.converter
        )
        backbone_model = converter(backbone_model)

    backbone_model.to(device)

    # Load segmentation model
    segmentation_model = build_segmentation_decoder(
        backbone_model=backbone_model,
        autocast_dtype=torch.float32,
        **cfg.segmentation_decoder
    ).to(device)

    # Build optimizer and scheduler
    if cfg.segmentation_decoder.tuning:
        base_lr = cfg.optimizer.lr
        backbone_lr = base_lr * 0.1

        backbone = segmentation_model.segmentation_model[0]
        decoder = segmentation_model.segmentation_model[1]

        optimizer = torch.optim.AdamW(
            [
                {"params": filter(lambda p: p.requires_grad, backbone.parameters()),
                 "lr": backbone_lr},
                {"params": filter(lambda p: p.requires_grad, decoder.parameters()),
                 "lr": base_lr},
            ],
            betas=cfg.optimizer.betas,
            weight_decay=cfg.optimizer.weight_decay,
        )

    else:
        optimizer = torch.optim.AdamW(
            params=filter(lambda p: p.requires_grad, segmentation_model.parameters()),
            **cfg.optimizer
        )
    scheduler = build_scheduler(
        optimizer=optimizer,
        lr=cfg.optimizer.lr,
        **cfg.scheduler
    )

    # Define loss
    criterion = MultiSegmentationLoss(
        **cfg.loss_function
    )

    # Define scaler
    scaler = torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu")

    total_iter = cfg.scheduler.total_iter
    global_step = 0
    best_metric = -1
    best_metrics = None
    best_iter = -1
    while global_step < total_iter:
        for i, batch in enumerate(loader_tr):
            if global_step >= total_iter:
                break

            segmentation_model.train()
            print(f"[*INFO*] Training iter {global_step + 1}/{total_iter}.")
            loss = train_step(
                segmentation_model=segmentation_model,
                batch=batch,
                device=device,
                scaler=scaler,
                optimizer=optimizer,
                optimizer_gradient_clip=cfg.gradient_clip,
                scheduler=scheduler,
                criterion=criterion,
                model_dtype=torch.float32,
            )

            if global_step % cfg.eval_interval == 0:
                print(f"[*INFO*] Evaluation iter {global_step + 1}/{total_iter}.")
                metrics = validate(
                    segmentation_model=segmentation_model,
                    val_loader=loader_ts,
                    criterion=criterion,
                    device=device,
                    cfg=cfg
                )
                if metrics[cfg.metric_to_save] > best_metric:
                    best_metric = metrics[cfg.metric_to_save]
                    best_metrics = metrics
                    best_iter = global_step
                    print(f"[*INFO*] Iteration {best_iter} evaluates the best metric: {best_metrics}.\n"
                          f"[*INFO*] Save the best model.")
                    os.makedirs(os.path.join(cfg.model_dir_to_save, f"{cfg.seed}"), exist_ok=True)
                    torch.save(
                        {
                            "model": segmentation_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        },
                        os.path.join(os.path.join(cfg.model_dir_to_save, f"{cfg.seed}"), f"best_model.pth")
                    )

            wandb.log(
                {'Global Step': global_step, 'Learning Rate': optimizer.param_groups[0]["lr"], 'Training Loss': loss,
                 'Test Loss': metrics['loss'], 'Test Dice': metrics['dice'], 'Test mIoU': metrics['mIoU']})
            global_step += 1

    print(f"[*INFO*] Last iter evaluation.")
    metrics = validate(
        segmentation_model=segmentation_model,
        val_loader=loader_ts,
        criterion=criterion,
        device=device,
        cfg=cfg
    )
    if metrics[cfg.metric_to_save] > best_metric:
        best_metric = metrics[cfg.metric_to_save]
        best_metrics = metrics
        print(f"[*INFO*] Last iter evaluates the best metric: {best_metrics}.\n"
              f"[*INFO*] Save the best model.")
        os.makedirs(os.path.join(cfg.model_dir_to_save, f"{cfg.seed}"), exist_ok=True)
        torch.save(
            {
                "model": segmentation_model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            os.path.join(os.path.join(cfg.model_dir_to_save, f"{cfg.seed}", f"best_model_last.pth"))
        )
    print(f"[*INFO*] Best metrics: {best_metrics}.\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    main(cfg)
