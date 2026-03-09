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
import torch.utils.data as data
import wandb
from PIL import Image
from medmnist import INFO
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision.transforms import v2
from tqdm import tqdm, trange

from planecycle.converters.dinov3_converter import PlaneCycleConverter

MODEL_WEIGHTS_MAP = {
    "dinov3_vits16": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "dinov3_vits16plus": "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
    "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "dinov3_vith16plus": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
    "dinov3_vit7b16": "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
}

def set_rng_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(gpu_ids: str) -> torch.device:
    visible_gpu_ids = [int(gid) for gid in gpu_ids.split(',') if int(gid) >= 0]
    if visible_gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_gpu_ids[0])
    device = torch.device(f'cuda:{visible_gpu_ids[0]}') if torch.cuda.is_available() else torch.device('cpu')
    print(f"[*] Running on device: {device}")
    return device


class Transform3D(nn.Module):
    def __init__(self, mode='val', resolution=64, target_resolution=64):
        super().__init__()
        self.mode = 'train' if 'train' in mode else 'val'
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1))

        need_resize = (target_resolution != resolution)
        pad_size = max(1, 4 * (target_resolution // resolution)) if need_resize else 4

        core_spatial_aug = [
            v2.RandomCrop(size=target_resolution, padding=pad_size, padding_mode='reflect'),
            v2.RandomHorizontalFlip(p=0.5), v2.RandomVerticalFlip(p=0.5),
            v2.RandomChoice([v2.RandomRotation(degrees=(90, 90)), v2.RandomRotation(degrees=(180, 180)),
                             v2.RandomRotation(degrees=(270, 270)), v2.Identity()]),
        ]

        if need_resize:
            self.train_aug = v2.Compose(
                [v2.Resize(size=target_resolution, interpolation=v2.InterpolationMode.BILINEAR), *core_spatial_aug])
            self.val_test_aug = v2.Resize(size=target_resolution, interpolation=v2.InterpolationMode.BILINEAR)
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
        x = self.train_aug(x) if self.mode == 'train' else self.val_test_aug(x)
        if x.shape[0] == 1:
            x = x.expand(3, -1, -1, -1)
        return (x - self.mean) / self.std


class Dinov3Linear(nn.Module):
    def __init__(self, *, backbone: nn.Module, embed_dim: int, D_slices: int, out_features: int,
                 final_pool_method: str = 'learn_to_pool', concat_patch_token: bool = False,
                 block_type="PlaneCycle"):
        super().__init__()
        self.backbone = backbone
        self.final_pool_method = final_pool_method
        self.concat_patch_token = concat_patch_token

        if self.final_pool_method not in {'mean', 'learn_to_pool', 'no_pool'}:
            raise ValueError(f"Unsupported final_pool_method: {self.final_pool_method!r}")

        head_dim = embed_dim * 2 if self.concat_patch_token else embed_dim
        if self.final_pool_method == "learn_to_pool":
            self.pool_head = nn.Linear(D_slices, 1)
        self.linear_head = nn.Linear(head_dim, out_features)
        self.block_type = block_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, depth, _, _ = x.shape
        features = self.backbone(x)
        cls_token = features["x_norm_clstoken"]
        if self.concat_patch_token:
            patch_tokens = features["x_norm_patchtokens"]
            cls_token = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
        if self.final_pool_method == 'no_pool' or self.block_type == 'Flatten3D':
            pooled = cls_token
        else:
            cls_signal = cls_token.reshape(-1, depth, cls_token.shape[1])
            if self.final_pool_method == 'mean':
                pooled = cls_signal.mean(dim=1)
            else:  # learn_to_pool
                pooled = self.pool_head(cls_signal.permute(0, 2, 1)).squeeze(-1)
        return self.linear_head(pooled)


def load_model(args, device, out_features):
    backbone = torch.hub.load(args.repo_path, args.arch, source='local', pretrained=False, block_type=args.block_type)

    weights_path = os.path.join(args.weight_dir, MODEL_WEIGHTS_MAP[args.arch])
    print(f"[*] Loading weights: {weights_path}")

    pretrained_weights = torch.load(weights_path, map_location=device)
    if args.block_type == "Flatten3D":
        unwanted_key = "rope_embed.periods"
        if unwanted_key in pretrained_weights:
            print(f"Removing {unwanted_key} from state_dict to avoid dimension mismatch.")
            del pretrained_weights[unwanted_key]

    backbone.load_state_dict(pretrained_weights, strict=True)

    if args.block_type == "PlaneCycle":
        backbone = PlaneCycleConverter(cycle_order=args.cycle_order, pool_method=args.pool_method)(backbone)

    model = Dinov3Linear(backbone=backbone, embed_dim=backbone.embed_dim, D_slices=args.D_slices,
                         out_features=out_features, final_pool_method=args.final_pool_method,
                         concat_patch_token=args.concat_patch_token,
                         block_type=args.block_type)
    print(model)

    if args.training_method == 'LP':
        for param in model.backbone.parameters():
            param.requires_grad = False

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"[*requires_grad*] {name}: {param.shape}")

    return model


def build_scheduler(args, optimizer):
    if args.scheduler == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.min_lr)
    if args.scheduler == "WarmupCosineAnnealingLR":
        warmup_scheduler = LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=args.warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(args.num_epochs - args.warmup_epochs),
                                             eta_min=args.min_lr)
        return SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs])
    milestones = [int(0.9 * args.num_epochs), int(0.95 * args.num_epochs)]
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)


def init_wandb_run(args):
    return wandb.init(
        entity=args.entity, project=args.project_name,
        name=args.run_name or f"{args.data_flag}_{args.arch}_{args.block_type}_{args.pool_method}",
        config={
            "dataset": args.data_flag,
            "architecture": args.arch, "block_type": args.block_type, "pool_method": args.pool_method,
            "final_pool_method": args.final_pool_method, "training_method": args.training_method,
            "cycle_order": args.cycle_order,
            "epochs": args.num_epochs, "batch_size": args.batch_size, "learning_rate": args.max_lr,
            "min_lr": args.min_lr, "weight_decay": args.weight_decay, "scheduler": args.scheduler,
            "warmup_epochs": args.warmup_epochs, "target_resolution": args.target_resolution,
            "size": args.size, "num_workers": args.num_workers, "seed": args.seed,
        },
    )


def train(model, train_loader, task, criterion, optimizer, device):
    total_loss = []
    model.train()
    for inputs, targets in tqdm(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        targets = targets.to(torch.float32).to(device) if task == 'multi-label, binary-class' else torch.squeeze(
            targets, 1).long().to(device)
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
            if task == 'multi-label, binary-class':
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
    output_root = os.path.join(args.output_root, args.data_flag, time.strftime("%y%m%d_%H%M%S"))
    os.makedirs(output_root, exist_ok=True)

    # Prepare data
    print('==> Preparing data...')
    info = INFO[args.data_flag]
    task, n_classes = info['task'], len(info['label'])
    data_class = getattr(medmnist, info['python_class'])

    train_transform = Transform3D(mode='train', resolution=args.size, target_resolution=args.target_resolution)
    eval_transform = Transform3D(mode='val', resolution=args.size, target_resolution=args.target_resolution)

    dataset_kwargs = dict(download=args.download, as_rgb=args.as_rgb, size=args.size)
    train_dataset = data_class(split='train', transform=train_transform, **dataset_kwargs)
    train_dataset_at_eval = data_class(split='train', transform=eval_transform, **dataset_kwargs)
    val_dataset = data_class(split='val', transform=eval_transform, **dataset_kwargs)
    test_dataset = data_class(split='test', transform=eval_transform, **dataset_kwargs)

    dl_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers)
    train_loader = data.DataLoader(train_dataset, shuffle=True, **dl_kwargs)
    train_loader_at_eval = data.DataLoader(train_dataset_at_eval, shuffle=False, **dl_kwargs)
    val_loader = data.DataLoader(val_dataset, shuffle=False, **dl_kwargs)
    test_loader = data.DataLoader(test_dataset, shuffle=False, **dl_kwargs)

    train_evaluator = medmnist.Evaluator(args.data_flag, 'train', size=args.size)
    val_evaluator = medmnist.Evaluator(args.data_flag, 'val', size=args.size)
    test_evaluator = medmnist.Evaluator(args.data_flag, 'test', size=args.size)

    criterion = nn.BCEWithLogitsLoss() if task == "multi-label, binary-class" else nn.CrossEntropyLoss()

    print('==> Building and training model...')
    model = load_model(args, device, n_classes).to(device)

    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path, map_location=device)['net'], strict=True)
        train_metrics = test(model, train_evaluator, train_loader_at_eval, task, criterion, device, args.run,
                             output_root)
        val_metrics = test(model, val_evaluator, val_loader, task, criterion, device, args.run, output_root)
        test_metrics = test(model, test_evaluator, test_loader, task, criterion, device, args.run, output_root)
        print(
            f'train  auc: {train_metrics[1]:.5f}  acc: {train_metrics[2]:.5f}\nval    auc: {val_metrics[1]:.5f}  acc: {val_metrics[2]:.5f}\ntest   auc: {test_metrics[1]:.5f}  acc: {test_metrics[2]:.5f}')

    if args.num_epochs == 0:
        return

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.max_lr,
                                  weight_decay=args.weight_decay)
    scheduler = build_scheduler(args, optimizer)

    logs = ['loss', 'auc', 'acc']
    train_logs = ['train_' + log for log in logs]
    val_logs = ['val_' + log for log in logs]
    test_logs = ['test_' + log for log in logs]
    log_dict = OrderedDict.fromkeys(train_logs + val_logs + test_logs, 0)

    best_auc, best_epoch, best_model = 0.0, 0, deepcopy(model)

    for epoch in trange(args.num_epochs):
        train_loss = train(model, train_loader, task, criterion, optimizer, device)
        train_metrics = test(model, train_evaluator, train_loader_at_eval, task, criterion, device, args.run)
        val_metrics = test(model, val_evaluator, val_loader, task, criterion, device, args.run)
        test_metrics = test(model, test_evaluator, test_loader, task, criterion, device, args.run)

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

        metric_names = ['loss', 'auc', 'acc']
        payload = {"epoch": epoch, "lr": lr}
        payload.update({f"train/{name}": val for name, val in zip(metric_names, train_metrics)})
        payload.update({f"val/{name}": val for name, val in zip(metric_names, val_metrics)})
        payload.update({f"test/{name}": val for name, val in zip(metric_names, test_metrics)})
        wandb.log(payload)

        cur_auc = val_metrics[1]
        if cur_auc >= best_auc:
            best_epoch, best_auc, best_model = epoch, cur_auc, deepcopy(model)
            print(f'cur_best_auc: {best_auc}, cur_best_epoch: {best_epoch}')
            wandb.run.summary.update({"best_auc": best_auc, "best_epoch": best_epoch})

    train_metrics = test(best_model, train_evaluator, train_loader_at_eval, task, criterion, device, args.run,
                         output_root)
    val_metrics = test(best_model, val_evaluator, val_loader, task, criterion, device, args.run, output_root)
    test_metrics = test(best_model, test_evaluator, test_loader, task, criterion, device, args.run, output_root)

    log = f"{args.data_flag}\ntrain  auc: {train_metrics[1]:.5f}  acc: {train_metrics[2]:.5f}\nval    auc: {val_metrics[1]:.5f}  acc: {val_metrics[2]:.5f}\ntest   auc: {test_metrics[1]:.5f}  acc: {test_metrics[2]:.5f}\n"
    print(log)

    wandb.run.summary.update({
        "final_train_auc": train_metrics[1], "final_train_acc": train_metrics[2],
        "final_val_auc": val_metrics[1], "final_val_acc": val_metrics[2],
        "final_test_auc": test_metrics[1], "final_test_acc": test_metrics[2],
    })

    with open(os.path.join(output_root, f'{args.data_flag}_log.txt'), 'a') as f:
        f.write(log)


def build_parser():
    parser = argparse.ArgumentParser(description='Train and evaluate a single-task MedMNIST 3D model.')

    # Experiment arguments
    exp = parser.add_argument_group("experiment")
    exp.add_argument('--project_name', default='dinov3', type=str, help='Weights & Biases project name.')
    exp.add_argument('--entity', default='<your_wandb_entity>', type=str, help='Weights & Biases entity or team name.')
    exp.add_argument('--run', default='model1', type=str, help='Suffix used by MedMNIST evaluator output files.')
    exp.add_argument('--run_name', default=None, type=str, help='Optional explicit W&B run name.')
    exp.add_argument('--output_root', default='./outputs', type=str, help='Directory for logs and evaluation outputs.')
    exp.add_argument('--gpu_ids', default='0', type=str, help='Comma-separated GPU ids, e.g. "0" or "0,1".')
    exp.add_argument('--num_workers', default=0, type=int, help='Number of DataLoader worker processes.')
    exp.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility.')
    exp.add_argument('--download', action='store_true', help='Download MedMNIST data if not found locally.')

    # Dataset arguments
    dset = parser.add_argument_group("dataset")
    dset.add_argument('--data_flag', default='organmnist3d', type=str, help='MedMNIST dataset name.')
    dset.add_argument('--size', default=64, type=int, help='Original dataset image size.')
    dset.add_argument('--target_resolution', default=64, type=int,
                      help='Target spatial resolution after preprocessing.')
    dset.add_argument('--batch_size', default=32, type=int, help='Mini-batch size for training and evaluation.')
    dset.add_argument('--as_rgb', action='store_true', help='Repeat single-channel volume to 3 channels.')

    # Optimization arguments
    opt = parser.add_argument_group("optimization")
    opt.add_argument('--num_epochs', default=100, type=int, help='Number of training epochs.')
    opt.add_argument('--max_lr', default=1e-3, type=float, help='Initial learning rate.')
    opt.add_argument('--min_lr', default=1e-6, type=float, help='Minimum learning rate for cosine schedulers.')
    opt.add_argument('--weight_decay', default=1e-2, type=float, help='Weight decay for AdamW.')
    opt.add_argument('--warmup_epochs', default=10, type=int, help='Number of warmup epochs.')
    opt.add_argument('--scheduler', default='WarmupCosineAnnealingLR', type=str,
                     help='Learning-rate scheduler: MultiStepLR, CosineAnnealingLR, or WarmupCosineAnnealingLR.')

    # Model arguments
    mdl = parser.add_argument_group("model")
    mdl.add_argument('--training_method', default='LP', type=str,
                     help='Training mode: LP(linear probing) or FT(finetune).')
    mdl.add_argument('--repo_path', default='models', type=str, help='Local torch.hub repository path.')
    mdl.add_argument('--weight_dir', default=None, type=str, help='Path to pretrained backbone weights.')
    mdl.add_argument('--arch', default='dinov3_vits16', type=str, help='Backbone architecture name.')
    mdl.add_argument('--block_type', default='PlaneCycle', type=str, help='Backbone block type or "PlaneCycle".')
    mdl.add_argument('--pool_method', default='PCg', type=str, help='PlaneCycle pooling method, PCg or PCm.')
    mdl.add_argument('--final_pool_method', default='learn_to_pool', type=str,
                     help='Final pooling method: mean, learn_to_pool, or no_pool.')
    mdl.add_argument('--D_slices', default=64, type=int, help='Number of depth slices for final pooling head.')
    mdl.add_argument('--concat_patch_token', action='store_true', help='Concatenate mean patch token to CLS token.')
    mdl.add_argument('--cycle_order', nargs='+', choices=['HW', 'DW', 'DH'], default=['HW', 'DW', 'DH', 'HW'],
                     help='Plane traversal order for PlaneCycle blocks.')

    # Evaluation arguments
    eva = parser.add_argument_group("evaluation")
    eva.add_argument('--model_path', default=None, type=str,
                     help='Optional checkpoint path for evaluation or warm start.')

    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    for key, value in vars(args).items():
        print(f"{key:20}: {value}")

    wandb_run = init_wandb_run(args)
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

    main(args)

    wandb_run.finish()
