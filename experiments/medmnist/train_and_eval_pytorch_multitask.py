import argparse
import os
import random
import time
from collections import OrderedDict
from copy import deepcopy

import matplotlib.pyplot as plt
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
from tqdm import tqdm
from tqdm import trange

from experiments.medmnist.config import MODEL_WEIGHTS_MAP, ROOT_DIR, PLANECYCLE
from models.hub.classification import Dinov3Linear
from planecycle.converters.dinov3_converter import PlaneCycleConverter


def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def show_plt(x):
    x_numpy = x[0, :, :, :].cpu().numpy()
    center_idx = x_numpy.shape[0] // 2
    center_frame = x_numpy[center_idx]

    # 4. 可视化
    plt.figure(figsize=(6, 6))
    # plt.imshow 不需要你手动将数据归一化到 0-255
    plt.imshow(center_frame, cmap='gray')  # 医疗影像通常使用灰度显示
    plt.title(f"Center Frame (Index: {center_idx})")
    plt.axis('off')
    plt.show()


class Transform3D(nn.Module):
    def __init__(self, mode='val', resolution=64, target_resolution=64, operator=''):
        super().__init__()
        self.mode = 'train' if 'train' in mode else 'val'
        self.operator = operator

        # 注册 Buffer：注意这里假设 C 在 expand 后为 3
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1))

        # 1. 逻辑判断
        need_resize = (target_resolution != resolution)
        # 即使不需要 resize，保持 4 像素的抖动也是好的
        pad_size = max(1, 4 * (target_resolution // resolution)) if need_resize else 4

        core_spatial_aug = [
            # 合并 Padding 和 Crop
            v2.RandomCrop(size=target_resolution, padding=pad_size, padding_mode='reflect'),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomChoice([
                v2.RandomRotation(degrees=(90, 90)),
                v2.RandomRotation(degrees=(180, 180)),
                v2.RandomRotation(degrees=(270, 270)),
                v2.Identity(),
            ]),
        ]

        if need_resize:
            self.train_aug = v2.Compose([
                v2.Resize(size=target_resolution, interpolation=v2.InterpolationMode.BILINEAR),
                *core_spatial_aug
            ])
            self.val_test_aug = v2.Resize(size=target_resolution, interpolation=v2.InterpolationMode.BILINEAR)
        else:
            self.train_aug = v2.Compose(core_spatial_aug)
            self.val_test_aug = v2.Identity()

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            if isinstance(x, Image.Image):  # 判断是否为 PIL 对象
                x = np.array(x)

                # 此时 x 已经是 Numpy 数组了，可以安全转换
            x = torch.from_numpy(x).float()

        # --- 新增：自动检测并归一化 ---
        # 如果数据最大值远大于 1（比如到了 255 或者 HU 值几百）
        # 且你打算使用 ImageNet 的 mean/std
        if x.max() > 100:
            # 这里以简单的 0-255 为例，如果是 CT 请根据窗宽窗位逻辑修改
            x = x / 255

        # 保证 x 至少是 [C, D, H, W]
        if x.ndim == 3:
            x = x.unsqueeze(0)
        elif x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)

        # 1. 空间增强
        if self.mode == 'train':
            x = self.train_aug(x)
        else:
            x = self.val_test_aug(x)

        # 2. 转换为 3 通道 (适配 ImageNet 预训练模型权重)
        if x.shape[0] == 1:
            x = x.expand(3, -1, -1, -1)

        # 3. 归一化 (减均值除标准差)
        # 注意：如果输入是 float，确保其范围在 0-1 之间，或者调整 mean/std
        x = (x - self.mean) / self.std

        # show_plt(x)

        return x


# class Transform3D(nn.Module):
#     def __init__(self, mode='val', resolution=64, target_resolution=64, operator=''):
#         super().__init__()
#         self.mode = 'train' if 'train' in mode else 'val'
#         self.target_res = target_resolution
#         self.need_resize = (target_resolution != resolution)
#
#         # 注册 Buffer
#         self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1))
#         self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1))
#
#         # 训练集的随机裁剪逻辑
#         # 即使 Resize 了，先算好 pad 再 Crop 也是高效的
#         pad_size = max(1, 4 * (target_resolution // resolution)) if self.need_resize else 4
#         self.random_crop = v2.RandomCrop(size=target_resolution, padding=pad_size, padding_mode='reflect')
#
#     def forward(self, x):
#         # x 输入 shape: (1, D, H, W)
#         if not isinstance(x, torch.Tensor):
#             x = torch.from_numpy(x).float()
#
#         # 1. 快速归一化：避免动态 max() 遍历，如果已知是 255 则直接除
#         # 如果必须动态，则保持 x = x / x.max()
#         if x.max() > 1.0:
#             x = x / x.max()
#
#         # 2. 插值优化：只针对 HW 平面
#         # 由于输入是 (1, D, H, W)，interpolate 会自动将 D 视为通道，只对最后两个维度插值
#         if self.need_resize:
#             x = F.interpolate(
#                 x,
#                 size=(self.target_res, self.target_res),
#                 mode='bilinear',
#                 align_corners=False
#             )
#
#         # 3. 空间增强 (Train 模式)
#         if self.mode == 'train':
#             # A. 随机裁剪
#             x = self.random_crop(x)
#             # B. 随机翻转 (极快)
#             if torch.rand(1) < 0.5: x = x.flip(-1)  # Horizontal
#             if torch.rand(1) < 0.5: x = x.flip(-2)  # Vertical
#             # C. 90度旋转 (极快，无插值开销)
#             k = torch.randint(0, 4, (1,)).item()
#             if k > 0:
#                 x = torch.rot90(x, k, dims=[-2, -1])
#
#         # 4. 转换为 3 通道 (先插值再 expand，速度最快)
#         if x.shape[0] == 1:
#             x = x.expand(3, -1, -1, -1)
#
#         # show_plt(x)
#
#         # 5. 归一化
#         return (x - self.mean) / self.std


# class Transform3D(nn.Module):
#     def __init__(self, mode='val', resolution=64, target_resolution=64, operator=''):
#         super().__init__()
#         self.mode = 'train' if 'train' in mode else 'val'
#         self.operator = operator
#
#         # 注册 Buffer：注意这里假设 C 在 expand 后为 3
#         self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1))
#         self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1))
#
#         # 1. 逻辑判断
#         need_resize = (target_resolution != resolution)
#         # 即使不需要 resize，保持 4 像素的抖动也是好的
#         pad_size = max(1, 4 * (target_resolution // resolution)) if need_resize else 4
#
#         core_spatial_aug = [
#             # 合并 Padding 和 Crop
#             v2.RandomCrop(size=target_resolution, padding=pad_size, padding_mode='reflect'),
#             v2.RandomHorizontalFlip(p=0.5),
#             v2.RandomVerticalFlip(p=0.5),
#             v2.RandomChoice([
#                 v2.RandomRotation(degrees=(90, 90)),
#                 v2.RandomRotation(degrees=(180, 180)),
#                 v2.RandomRotation(degrees=(270, 270)),
#                 v2.Identity(),
#             ]),
#         ]
#
#         if need_resize:
#             self.train_aug = v2.Compose([
#                 v2.Resize(size=target_resolution, interpolation=v2.InterpolationMode.BILINEAR),
#                 *core_spatial_aug
#             ])
#             self.val_test_aug = v2.Resize(size=target_resolution, interpolation=v2.InterpolationMode.BILINEAR)
#         else:
#             self.train_aug = v2.Compose(core_spatial_aug)
#             self.val_test_aug = v2.Identity()
#
#     def forward(self, x):
#         if not isinstance(x, torch.Tensor):
#             x = torch.from_numpy(x).float()
#
#         # --- 新增：自动检测并归一化 ---
#         # 如果数据最大值远大于 1（比如到了 255 或者 HU 值几百）
#         # 且你打算使用 ImageNet 的 mean/std
#         if x.max() > 1.0:
#             # 这里以简单的 0-255 为例，如果是 CT 请根据窗宽窗位逻辑修改
#             x = x / x.max()
#
#         # 保证 x 至少是 [C, D, H, W]
#         if x.ndim == 3:
#             x = x.unsqueeze(0)
#
#         # 1. 空间增强
#         if self.mode == 'train':
#             x = self.train_aug(x)
#         else:
#             x = self.val_test_aug(x)
#
#         # 2. 转换为 3 通道 (适配 ImageNet 预训练模型权重)
#         if x.shape[0] == 1:
#             x = x.expand(3, -1, -1, -1)
#
#         # 3. 归一化 (减均值除标准差)
#         # 注意：如果输入是 float，确保其范围在 0-1 之间，或者调整 mean/std
#         x = (x - self.mean) / self.std
#
#         # show_plt(x)
#
#         return x


# class Transform3D(nn.Module):
#     def __init__(self, mode='val', resolution=64, target_resolution=256, operator=''):
#         super().__init__()
#         if target_resolution > resolution:
#             self.need_resize = True
#         else:
#             self.need_resize = False
#
#         self.mode = 'train' if 'train' in mode else 'val'
#         self.operator = operator
#
#         # 预先转为 Tensor 减少运行时开销
#         self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1))
#         self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1))
#
#         # 1. 纯几何变换（不改变尺寸/像素精度）
#         geometric_aug = [
#             v2.Pad(padding=4 * (target_resolution // resolution), padding_mode='reflect'),
#             v2.RandomCrop(size=target_resolution),
#             v2.RandomHorizontalFlip(p=0.5),
#             v2.RandomVerticalFlip(p=0.5),
#             v2.RandomChoice([
#                 v2.RandomRotation(degrees=(90, 90)),
#                 v2.RandomRotation(degrees=(180, 180)),
#                 v2.RandomRotation(degrees=(270, 270)),
#                 v2.Identity(),
#             ]),
#         ]
#
#         if self.need_resize is True:
#             # # 路径 A：缩放 + 几何变换
#             # # 这个做法有点问题：训练的结节相比val的结节都偏大的
#             # self.train_aug = v2.Compose([
#             #     v2.RandomResizedCrop(size=target_resolution, scale=(0.8, 1.0), ratio=(1, 1)),
#             #     *geometric_aug
#             # ])
#             # self.val_test_aug = v2.Resize(size=target_resolution)
#             # 路径 A 改进：先统一放大，再进行空间增强
#             self.train_aug = v2.Compose([
#                 # 1. 先放大到一个略大于 256 的尺寸，比如 280
#                 v2.Resize(size=target_resolution, interpolation=v2.InterpolationMode.BICUBIC),
#                 # 2. 在大图上做裁剪，减少插值带来的信息损失感
#                 *geometric_aug
#             ])
#             # 验证集也需要 Resize 到相同尺寸
#             self.val_test_aug = v2.Resize(size=target_resolution, interpolation=v2.InterpolationMode.BICUBIC)
#         else:
#             # 路径 B：无损位移 + 几何变换
#             self.train_aug = v2.Compose([
#                 *geometric_aug
#             ])
#             self.val_test_aug = v2.Identity()
#
#     def forward(self, x):
#         if not isinstance(x, torch.Tensor):
#             x = torch.from_numpy(x).float()
#         # x.shape [1, D, H, W]
#
#         # 1. 空间增强 (v2 支持 [C, D, H, W]，会自动对最后两维操作)
#         # if self.resize_2target is True:
#         if self.mode == 'train':
#             x = self.train_aug(x)
#         else:
#             x = self.val_test_aug(x)
#
#         x = x.expand(3, -1, -1, -1)
#
#         # 直接在 Tensor 上计算，不调用 normalize 变换函数，减少封装开销
#         x = (x - self.mean) / self.std
#
#         # show_plt(x)
#         # print(f"shape after transform: {x.shape}")
#         return x


def load_model(args, device, out_features):
    backbone = torch.hub.load(
        args.repo_path,
        args.arch,
        source='local',
        pretrained=False,
        block_type=args.block_type,
    )
    if args.train_from_scratch is False:
        weights_path = os.path.join(args.weights_dir, MODEL_WEIGHTS_MAP[args.arch])
        print(f"[*] 正在载入权重: {os.path.basename(weights_path)}")
        pretrained_weights = torch.load(weights_path, map_location=device)

        if args.use_universal_rope:
            unwanted_key = "rope_embed.periods"
            if unwanted_key in pretrained_weights:
                print(f"Removing {unwanted_key} from state_dict to avoid dimension mismatch.")
                del pretrained_weights[unwanted_key]

        backbone.load_state_dict(pretrained_weights, strict=True)
    else:
        print(f"model train from scratch mode: {args.train_from_scratch}")

    # 1. 实例化模型
    embed_dim = backbone.embed_dim
    if args.block_type == PLANECYCLE:
        backbone = PlaneCycleConverter(
            cycle_order=args.cycle_order,
            pool_method=args.pool_method,
        )(backbone)
    model = Dinov3Linear(backbone=backbone, embed_dim=embed_dim, D_slices=args.D_slices, out_features=out_features,
                         final_pool_method=args.final_pool_method,
                         concat_patch_token=args.concat_patch_token,
                         dataset_type=args.dataset_type)
    print(model)
    # 2. 冻结 Backbone (Linear Probing 的核心步骤)
    if args.training_method == 'LP':
        for param in model.backbone.parameters():  # 会递归所有子模块
            param.requires_grad = False

    for name, param in model.named_parameters():
        if param.requires_grad is True:
            print(f"[*requires_grad*:] {name}, {param.shape}")

    return model


class EarlyStoppingAUC:
    """当验证集 AUC 在给定的耐心值内不再改善时，停止训练。"""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): 上次 AUC 改善后等待的 epoch 数。默认: 7
            verbose (bool): 如果为 True，为每次 AUC 改善打印一条信息。默认: False
            delta (float): 监测指标的最小变化量（只有增加量超过 delta 才算改善）。默认: 0
            path (str): 最佳模型保存的路径。默认: 'best_model.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_auc):
        # AUC 越高越好，所以 score 就是 auc 本身
        score = val_auc

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            # 如果当前 AUC 没有超过之前的最高分（加上一个微小的 delta）
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # AUC 改善了
            self.best_score = score
            self.counter = 0


class EarlyStoppingLoss:
    """当验证集 AUC 在给定的耐心值内不再改善时，停止训练。"""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): 上次 AUC 改善后等待的 epoch 数。默认: 7
            verbose (bool): 如果为 True，为每次 AUC 改善打印一条信息。默认: False
            delta (float): 监测指标的最小变化量（只有增加量超过 delta 才算改善）。默认: 0
            path (str): 最佳模型保存的路径。默认: 'best_model.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        # AUC 越高越好，所以 score 就是 auc 本身
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            # 如果当前 AUC 没有超过之前的最高分（加上一个微小的 delta）
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # AUC 改善了
            self.best_score = score
            self.counter = 0


def main(data_flag, output_root, num_epochs, gpu_ids, batch_size, size, conv, pretrained_3d, download, model_flag,
         as_rgb, shape_transform, model_path, run, args):
    info = INFO[data_flag]
    task = info['task']
    n_channels = 3 if as_rgb else info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

    # get device
    str_ids = gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])

    device = torch.device('cuda:{}'.format(gpu_ids[0])) if torch.cuda.is_available() else torch.device('cpu')
    print(f"[*] 运行设备: {device}")

    output_root = os.path.join(output_root, data_flag, time.strftime("%y%m%d_%H%M%S"))
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    print('==> Preparing data...')

    train_transform = Transform3D(mode='train', resolution=args.size, target_resolution=args.target_resolution,
                                  operator=args.operator)
    eval_transform = Transform3D(mode='val', resolution=args.size, target_resolution=args.target_resolution,
                                 operator=args.operator)

    train_dataset = DataClass(split='train', transform=train_transform, download=download, as_rgb=as_rgb, size=size)
    train_dataset_at_eval = DataClass(split='train', transform=eval_transform, download=download, as_rgb=as_rgb,
                                      size=size)
    val_dataset = DataClass(split='val', transform=eval_transform, download=download, as_rgb=as_rgb, size=size)
    test_dataset = DataClass(split='test', transform=eval_transform, download=download, as_rgb=as_rgb, size=size)

    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   num_workers=args.num_workers,
                                   shuffle=True)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset_at_eval,
                                           batch_size=batch_size,
                                           num_workers=args.num_workers,
                                           shuffle=False)
    val_loader = data.DataLoader(dataset=val_dataset,
                                 batch_size=batch_size,
                                 num_workers=args.num_workers,
                                 shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=False)

    print('==> Building and training model...')

    model = load_model(args, device, out_features=n_classes)

    model = model.to(device)

    train_evaluator = medmnist.Evaluator(data_flag, 'train', size=size)
    val_evaluator = medmnist.Evaluator(data_flag, 'val', size=size)
    test_evaluator = medmnist.Evaluator(data_flag, 'test', size=size)

    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device)['net'], strict=True)
        train_metrics = test(model, train_evaluator, train_loader_at_eval, task, criterion, device, run, output_root)
        val_metrics = test(model, val_evaluator, val_loader, task, criterion, device, run, output_root)
        test_metrics = test(model, test_evaluator, test_loader, task, criterion, device, run, output_root)

        print('train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2]) + \
              'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2]) + \
              'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2]))

    if num_epochs == 0:
        return

    # 优化器只传入需要梯度的参数 (推荐做法)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.max_lr,
                                  weight_decay=args.weight_decay)

    if args.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=args.min_lr)
    elif args.scheduler == "WarmupCosineAnnealingLR":
        warmup_epochs = args.warmup_epochs  # 前 5 个 epoch 进行预热
        warmup_scheduler = LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(num_epochs - warmup_epochs), eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    else:
        gamma = 0.1
        milestones = [0.9 * num_epochs, 0.95 * num_epochs]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    # early_stoping_param = args.early_stoping_param
    # if early_stoping_param == 'auc':
    #     early_stopping = EarlyStoppingAUC(patience=args.stop_patience, verbose=True)
    # elif early_stoping_param == 'loss':
    #     early_stopping = EarlyStoppingLoss(patience=args.stop_patience, verbose=True)
    # else:
    #     early_stopping = EarlyStoppingAUC(patience=args.stop_patience, verbose=True)

    logs = ['loss', 'auc', 'acc']
    train_logs = ['train_' + log for log in logs]
    val_logs = ['val_' + log for log in logs]
    test_logs = ['test_' + log for log in logs]
    log_dict = OrderedDict.fromkeys(train_logs + val_logs + test_logs, 0)

    best_auc = 0
    best_epoch = 0
    best_model = deepcopy(model)

    global iteration
    iteration = 0

    for epoch in trange(num_epochs):
        train_loss = train(model, train_loader, task, criterion, optimizer, device)
        # train_metrics = test(model, train_evaluator, train_loader_at_eval, criterion, device, run)
        train_metrics = [train_loss, 0, 0]
        val_metrics = test(model, val_evaluator, val_loader, task, criterion, device, run)
        test_metrics = test(model, test_evaluator, test_loader, task, criterion, device, run)

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

        metrics_names = ['loss', 'auc', 'acc']
        # 2. 构造 log 字典
        # 使用 "分组/指标" 的命名方式
        payload = {"epoch": epoch, "lr": lr}

        # 动态填充 train, val, test 指标
        payload.update({f"train/{name}": val for name, val in zip(metrics_names, train_metrics)})
        payload.update({f"val/{name}": val for name, val in zip(metrics_names, val_metrics)})
        payload.update({f"test/{name}": val for name, val in zip(metrics_names, test_metrics)})

        # 3. 一次性上传
        wandb.log(payload)

        cur_auc = val_metrics[1]
        if cur_auc >= best_auc:
            best_epoch = epoch
            best_auc = cur_auc
            best_model = deepcopy(model)

            print('cur_best_auc:', best_auc)
            print('cur_best_epoch', best_epoch)
            # 核心：更新 W&B 的汇总统计
            wandb.run.summary["best_auc"] = best_auc
            wandb.run.summary["best_epoch"] = best_epoch

            state = {
                'net': model.state_dict(),
            }

            # path = os.path.join(output_root, 'best_model.pth')
            # torch.save(state, path)

        # # 调用早停逻辑
        # early_stopping(cur_auc)
        #
        # if early_stopping.early_stop:
        #     print("Early stopping triggered. Training finished.")
        #     break

    train_metrics = test(best_model, train_evaluator, train_loader_at_eval, criterion, device, run, output_root)
    val_metrics = test(best_model, val_evaluator, val_loader, criterion, device, run, output_root)
    test_metrics = test(best_model, test_evaluator, test_loader, criterion, device, run, output_root)

    train_log = 'train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2])
    val_log = 'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2])
    test_log = 'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2])

    log = '%s\n' % (data_flag) + train_log + val_log + test_log + '\n'
    print(log)

    # 2. 同步到 W&B Summary
    # 这样在 W&B 的表格列里就能直接看到这几个最终数值
    wandb.run.summary["final_train_auc"] = train_metrics[1]
    wandb.run.summary["final_train_acc"] = train_metrics[2]

    wandb.run.summary["final_val_auc"] = val_metrics[1]
    wandb.run.summary["final_val_acc"] = val_metrics[2]

    wandb.run.summary["final_test_auc"] = test_metrics[1]
    wandb.run.summary["final_test_acc"] = test_metrics[2]

    with open(os.path.join(output_root, '%s_log.txt' % (data_flag)), 'a') as f:
        f.write(log)


def train(model, train_loader, task, criterion, optimizer, device):
    total_loss = []
    global iteration

    model.train()
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32).to(device)
            loss = criterion(outputs, targets)
        else:
            targets = torch.squeeze(targets, 1).long().to(device)
            loss = criterion(outputs, targets)

        total_loss.append(loss.item())
        # print('train_loss_logs', loss.item(), iteration)
        iteration += 1

        loss.backward()
        optimizer.step()

    epoch_loss = sum(total_loss) / len(total_loss)
    return epoch_loss


def test(model, evaluator, data_loader, task, criterion, device, run, save_folder=None):

    model.eval()

    total_loss = []
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate((tqdm(data_loader))):
            outputs = model(inputs.to(device))

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                loss = criterion(outputs, targets)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = torch.squeeze(targets, 1).long().to(device)
                loss = criterion(outputs, targets)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            total_loss.append(loss.item())

            y_score = torch.cat((y_score, outputs), 0)

        y_score = y_score.detach().cpu().numpy()
        auc, acc = evaluator.evaluate(y_score, save_folder, run)

        test_loss = sum(total_loss) / len(total_loss)

        return [test_loss, auc, acc]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RUN Baseline model of MedMNIST3D')
    parser.add_argument('--data_flag',
                        default='organmnist3d',
                        type=str)
    parser.add_argument('--other_flag',
                        default='',
                        type=str)
    parser.add_argument('--project_name',
                        default='dinov3',
                        type=str)
    parser.add_argument('--output_root',
                        default='/scratch/work/yuy10/DINOv3/output_dir',
                        help='output root, where to save models',
                        type=str)
    # 模型相关参数
    parser.add_argument('--training_method', type=str, default='LP',
                        help='linear probing or finetune')
    parser.add_argument('--repo_path', type=str, default=os.path.join(ROOT_DIR, 'PlaneCycle', 'models'),
                        help='模型代码本地仓库路径')
    parser.add_argument('--weights_dir', type=str, default=os.path.join(ROOT_DIR, 'model_weights'),
                        help='模型权重文件路径')
    parser.add_argument('--arch', type=str, default='dinov3_vits16', help='模型架构名称')
    parser.add_argument('--dataset_type', type=str, default='3D', help='2D, 3D, or 2D_3D')
    parser.add_argument("--max_lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument("--warmup_epochs", default=10, type=int)
    parser.add_argument("--min_lr", default=1e-6, type=float)
    # 推理配置参数
    parser.add_argument(
        '--cycle_order',
        nargs='+',
        choices=['HW', 'DW', 'DH'],  # 限制输入范围
        default=['HW', 'DW', 'DH', 'HW'],  # 设置默认列表
        help="传入列表，例如: --cycle_order HW DW DH HW"
    )
    parser.add_argument('--block_type', type=str, default='cycle', help='DINO block type (e.g., cycle, original)')
    parser.add_argument('--pool_method', type=str, default='mean', help='Pooling method (e.g., mean, None)')
    parser.add_argument('--final_pool_method', type=str, default='mean', help='Pooling method (e.g., mean, None)')
    parser.add_argument('--early_stoping_param', type=str, default='auc', help='auc or loss')
    # parser.add_argument('--target_res', type=int, default=256, help='输入模型的前处理分辨率')
    parser.add_argument('--num_workers', type=int, default=0, help='workers')
    parser.add_argument("--scheduler", default="MultiStepL", type=str, help="scheduler method")
    parser.add_argument("--stop_patience", default=10, type=int, help="scheduler method")
    parser.add_argument("--seed", default=42, type=int, help="scheduler method")
    parser.add_argument("--D_slices", default=64, type=int, help="D_slices")
    parser.add_argument('--disable_rope', action="store_true")
    parser.add_argument('--use_universal_rope', action="store_true")
    # parser.add_argument('--flatten_3D', action="store_true")
    parser.add_argument('--rope_dim', type=int, default=2, help='auc or loss')
    parser.add_argument('--train_from_scratch', action="store_true")
    parser.add_argument('--num_epochs',
                        default=100,
                        help='num of epochs of training, the script would only test model if set num_epochs to 0',
                        type=int)
    parser.add_argument('--size',
                        default=64,
                        help='the image size of the dataset, 28 or 64, default=28',
                        type=int)
    parser.add_argument("--target_resolution", default=64, type=int, help="scheduler method")
    parser.add_argument('--gpu_ids',
                        default='0',
                        type=str)
    parser.add_argument('--batch_size',
                        default=32,
                        type=int)
    parser.add_argument('--operator',
                        default='',
                        help='choose converter from Conv2_5d, Conv3d, ACSConv',
                        type=str)
    parser.add_argument('--conv',
                        default='acs',
                        help='choose converter from Conv2_5d, Conv3d, ACSConv',
                        type=str)
    parser.add_argument('--pretrained_3d',
                        default='i3d',
                        type=str)
    parser.add_argument('--download',
                        action="store_true")
    parser.add_argument('--resize_2target',
                        action="store_true")
    parser.add_argument('--concat_patch_token', action="store_true")
    parser.add_argument('--as_rgb',
                        help='to copy channels, tranform shape 1x28x28x28 to 3x28x28x28',
                        action="store_true")
    parser.add_argument('--shape_transform',
                        help='for shape dataset, whether multiply 0.5 at eval',
                        action="store_true")
    parser.add_argument('--model_path',
                        default=None,
                        help='root of the pretrained model to test',
                        type=str)
    parser.add_argument('--model_flag',
                        default='resnet18',
                        help='choose backbone, resnet18/resnet50',
                        type=str)
    parser.add_argument('--run',
                        default='model1',
                        help='to name a standard evaluation csv file, named as {flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv',
                        type=str)

    # parser.add_argument('--train_method',
    #                     default='linear_probing',
    #                     type=str)

    args = parser.parse_args()
    args_dict = vars(args)
    for key, value in args_dict.items():
        print(f" {key:15}: {value}")  # 强制占用 15 个字符的宽度

    data_flag = args.data_flag
    output_root = args.output_root
    num_epochs = args.num_epochs
    size = args.size
    gpu_ids = args.gpu_ids
    batch_size = args.batch_size
    operator = args.operator
    conv = args.conv
    pretrained_3d = args.pretrained_3d
    download = args.download
    model_flag = args.model_flag
    as_rgb = args.as_rgb
    model_path = args.model_path
    shape_transform = args.shape_transform
    arch = args.arch
    block_type = args.block_type
    pool_method = args.pool_method
    max_lr = args.max_lr
    other_flag = args.other_flag
    final_pool_method = args.final_pool_method
    # train_method = args.train_method
    project_name = args.project_name
    resize_2target = args.resize_2target
    scheduler = args.scheduler
    run = args.run
    stop_patience = args.stop_patience
    concat_patch_token = args.concat_patch_token
    target_resolution = args.target_resolution
    seed = args.seed
    training_method = args.training_method
    cycle_order = args.cycle_order
    dataset_type = args.dataset_type

    # Start a new wandb run to track this script.
    wandbrun = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="yuyinghong1-aalto-university",
        # Set the wandb project where this run will be logged.
        project=project_name,
        name=f"{data_flag}_{block_type}_{pool_method}_{final_pool_method}_{resize_2target}",
        # Track hyperparameters and run metadata.
        config={
            "architecture": arch,
            "dataset": data_flag,
            "epochs": num_epochs,
            "learning_rate": max_lr,
            "batch_size": batch_size,
            "num_workers": args.num_workers,
            # "train_method": train_method,
            "resize_2target": resize_2target,
            "scheduler": scheduler,
            "stop_patience": stop_patience,
            "block_type": block_type,
            "pool_method": pool_method,
            "final_pool_method": final_pool_method,
            "concat_patch_token": concat_patch_token,
            "target_resolution": target_resolution,
            "seed": seed,
            'training_method': training_method,
            'cycle_order': cycle_order,
            "dataset_type": dataset_type,
        },
    )
    wandb.define_metric("epoch")  # 定义 epoch 为一个指标
    wandb.define_metric("*", step_metric="epoch")  # 设置以后所有指标默认以 epoch 为 x 轴

    set_rng_seed(seed=args.seed)

    main(data_flag, output_root, num_epochs, gpu_ids, batch_size, size, conv, pretrained_3d, download, model_flag,
         as_rgb, shape_transform, model_path, run, args)

    wandbrun.finish()
