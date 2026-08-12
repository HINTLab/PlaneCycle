import argparse
import os
import random
import sys
import time
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

# repo root on sys.path so dinov3/, planecycle/ and experiments/baselines/
# resolve when running this script directly from experiments/medmnist/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import medmnist
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import wandb
from medmnist import INFO
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm, trange

from config import MODEL_WEIGHTS_MAP
from loaders import load_ctfm_model, load_model, load_spectre_model
from transforms import Lifted2DTransform, Native3DTransform


def set_rng_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(gpu_ids: str) -> tuple[torch.device, list[int]]:
    requested = [int(gid) for gid in gpu_ids.split(",") if int(gid) >= 0]
    if not torch.cuda.is_available():
        print("[*] CUDA unavailable; running on CPU")
        return torch.device("cpu"), []
    if not requested:
        requested = [0]
    invalid = [gid for gid in requested if gid >= torch.cuda.device_count()]
    if invalid:
        raise ValueError(
            f"Requested GPU ids {invalid}, but only {torch.cuda.device_count()} "
            "CUDA devices are visible"
        )
    device = torch.device(f"cuda:{requested[0]}")
    print(f"[*] Running on GPU(s): {requested}; primary device: {device}")
    return device, requested


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
        name=args.run_name,  # None -> W&B auto-generates; the launcher passes one
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
            "channels_data": args.channels_data,
            "disable_converter": args.disable_converter,
            "D_slices": args.D_slices,
            "as_rgb": args.as_rgb,
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
    device, gpu_ids = get_device(args.gpu_ids)
    set_rng_seed(args.seed)
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
        train_transform = Native3DTransform(
            mode="train", resolution=args.size, target_resolution=args.target_resolution
        )
        eval_transform = Native3DTransform(
            mode="val", resolution=args.size, target_resolution=args.target_resolution
        )
        dataset_kwargs = dict(download=args.download, as_rgb=False, size=args.size)
    else:
        train_transform = Lifted2DTransform(
            mode="train", resolution=args.size, target_resolution=args.target_resolution
        )
        eval_transform = Lifted2DTransform(
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
        model = load_model(args, device, n_classes)

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

    if len(gpu_ids) > 1:
        model = nn.DataParallel(
            model, device_ids=gpu_ids, output_device=gpu_ids[0]
        )
        print(f"[*] Enabled nn.DataParallel on {len(gpu_ids)} GPUs: {gpu_ids}")

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
            # torch.save({"net": best_model.state_dict()},
            #            os.path.join(output_root, "best_model.pth"))

        # torch.cuda.empty_cache()

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
        help="Comma-separated CUDA device ids. One id uses a single GPU; "
        "multiple ids enable nn.DataParallel, e.g. --gpu_ids=0,1.",
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
        "--data_flag",
        default="organmnist3d",
        type=str,
        choices=[
            "organmnist3d",
            "nodulemnist3d",
            "adrenalmnist3d",
            "fracturemnist3d",
            "vesselmnist3d",
            "synapsemnist3d",
        ],
        help="MedMNIST 3D dataset name.",
    )
    dset.add_argument(
        "--size",
        default=64,
        type=int,
        choices=[28, 64],
        help="Original dataset image size.",
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
        help="Upsample H/W by this factor before the backbone (D unchanged).",
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
        choices=["MultiStepLR", "CosineAnnealingLR", "WarmupCosineAnnealingLR"],
        help="Learning-rate scheduler.",
    )

    # Model arguments
    mdl = parser.add_argument_group("model")
    mdl.add_argument(
        "--training_method",
        default="LP",
        type=str,
        choices=["LP", "FT"],
        help="LP (linear probing: backbone frozen) or FT (finetune).",
    )
    mdl.add_argument(
        "--channels_data",
        default="repeated",
        type=str,
        choices=["repeated", "neighbors"],
        help="Input channel encoding: repeated grayscale, or neighbouring "
        "slices (d-1, d, d+1). TriSlice implies neighbors.",
    )
    mdl.add_argument(
        "--weight_dir",
        default=None,
        type=str,
        help="Directory holding the pretrained backbone weights "
        "(filenames in config.py MODEL_WEIGHTS_MAP).",
    )
    mdl.add_argument(
        "--arch",
        default="dinov3_vits16",
        type=str,
        choices=sorted(MODEL_WEIGHTS_MAP),
        help="Backbone architecture (keys of config.py MODEL_WEIGHTS_MAP).",
    )
    mdl.add_argument(
        "--block_type",
        default="PlaneCycle",
        choices=["PlaneCycle", "Slice2D", "Flatten3D", "TriSlice", "ACS"],
        help="Method: PlaneCycle, Slice2D (2D), Flatten3D (3D), TriSlice (2.5D), "
        "ACS (native-3D ACSConv, ConvNeXt only).",
    )
    mdl.add_argument(
        "--disable_converter",
        action="store_true",
        help="PlaneCycle only: use BaselineViT/ConvNeXt's converter-equivalent "
        "mode instead of planecycle_converter (debug/ablation; outputs are "
        "equivalence-tested).",
    )
    mdl.add_argument(
        "--pool_method",
        default="PCg",
        choices=["PCg", "PCm"],
        help="PlaneCycle global-token pooling (ViT only).",
    )
    mdl.add_argument(
        "--final_pool_method",
        default="learn_to_pool",
        type=str,
        choices=["learn_to_pool", "mean"],
        help="Aggregation of per-slice CLS tokens into one embedding.",
    )
    mdl.add_argument(
        "--D_slices",
        default=64,
        type=int,
        help="Depth slices seen by learn_to_pool; must match the input D.",
    )
    mdl.add_argument(
        "--cycle_order",
        nargs="+",
        choices=["HW", "DW", "DH"],
        default=[],
        help="Plane traversal order for PlaneCycle blocks.",
    )

    # Comparison-baseline arguments (SPECTRE / CT-FM)
    spe = parser.add_argument_group("comparison baselines")
    spe.add_argument(
        "--model_family",
        default="dinov3",
        choices=["dinov3", "spectre", "ctfm"],
        help='Model family. "spectre"/"ctfm" switch to the natively-3D comparison backbones and Native3DTransform.',
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
