#!/bin/bash
#SBATCH --partition=gpu-h200-141g-ellis,gpu-h200-141g-short
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --array=0-11   # 大范围占位，脚本内部自动检查
#SBATCH -t 8:00:00
#SBATCH -J hybrid_1e-4
#SBATCH -o logs/med_%A_%a.out

mkdir -p logs
set -euo pipefail

echo "===== JOB START ====="
date
hostname

# --- 环境准备 ---
if [ -f "$WRKDIR/.bashrc.sh" ]; then
  source "$WRKDIR/.bashrc.sh"
elif [ -f ".bashrc.sh" ]; then
  source ".bashrc.sh"
fi

module --force purge
module load mamba

ENV_NAME="dinov3"
export ROOT_DIR="/scratch/work/yuy10/DINOv3/PlaneCycle"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR"
source activate "${ENV_NAME}"

# --- 配置 ---
#seeds=(42 123 1024 1337 2026)
seeds=(42)

datasets=(
  "chestmnist_nodulemnist3d_1e-4_lambda11"
#  "organmnist3d"
#  "adrenalmnist3d"
#  "fracturemnist3d"
#  "vesselmnist3d"
#  "synapsemnist3d"
)

arch_list=(
  "dinov3_vits16"
  "dinov3_vitb16"
  "dinov3_vitl16"
)

# 定义两种 cycle 模式
cycle_orders=(
  "HW DH DW"
  "HW DH DW HW"
)

final_pool_methods=("mean")

#configs=(
#  "cycle:group_tokens_mean"
#  "original:"
#  "cycle:mean"
#  "flatten_3D:"
#)

configs=(
  "PlaneCycle:PCg"
  "PlaneCycle:PCm"
)


resolutions=(64)

name="Hybrid_2D3D_FT"

# --- 自动计算总任务数 ---
total=$(( ${#seeds[@]} *
          ${#datasets[@]} *
          ${#arch_list[@]} *
          ${#final_pool_methods[@]} *
          ${#configs[@]} *
          ${#resolutions[@]} *
          ${#cycle_orders[@]} ))

echo "Total jobs: $total"
echo "Current SLURM ID: ${SLURM_ARRAY_TASK_ID:-0}"

if [[ ${SLURM_ARRAY_TASK_ID:-0} -ge $total ]]; then
  echo "Task ID exceeds total jobs — exiting."
  exit 0
fi

i=${SLURM_ARRAY_TASK_ID:-0}

# --- 解耦索引 ---
n_seed=${#seeds[@]}
n_data=${#datasets[@]}
n_arch=${#arch_list[@]}
n_fpool=${#final_pool_methods[@]}
n_conf=${#configs[@]}
n_res=${#resolutions[@]}
n_cycle=${#cycle_orders[@]}

cycle_idx=$(( i % n_cycle )); i=$(( i / n_cycle ))
res_idx=$(( i % n_res )); i=$(( i / n_res ))
conf_idx=$(( i % n_conf )); i=$(( i / n_conf ))
f_idx=$(( i % n_fpool )); i=$(( i / n_fpool ))
arch_idx=$(( i % n_arch )); i=$(( i / n_arch ))
data_idx=$(( i % n_data )); i=$(( i / n_data ))
seed_idx=$(( i % n_seed ))

# --- 取值 ---
target_cycle=${cycle_orders[$cycle_idx]}
target_seed=${seeds[$seed_idx]}
target_res=${resolutions[$res_idx]}
arch=${arch_list[$arch_idx]}
final_pool=${final_pool_methods[$f_idx]}
data_flag=${datasets[$data_idx]}
project_name="${arch}_${name}_${target_seed}"

current_config=${configs[$conf_idx]}
block_type=${current_config%:*}
pool_method=${current_config#*:}

# --- flatten_3D 特殊参数 ---
extra_args=""

case "$block_type" in
  flatten_3D)
    extra_args="--use_universal_rope --rope_dim=3"
    final_pool="no_pool"
    ;;
esac

if [[ "$block_type" == "flatten_3D" ]]; then
  echo "[INFO] flatten_3D detected → overriding final_pool to no_pool"
fi

# --- Debug banner ---
echo "===== RUN CONFIG ====="
echo "Dataset        : $data_flag"
echo "Arch           : $arch"
echo "Config         : $current_config"
echo "Block type     : $block_type"
echo "Pool method    : $pool_method"
echo "Final pool     : $final_pool"
echo "Resolution     : $target_res"
echo "Seed           : $target_seed"
echo "Extra args     : $extra_args"
echo "Cycle Order    : $target_cycle"
echo "======================"

# --- 运行 ---
python experiments/medmnist/train_and_eval_pytorch_multitask.py \
    --project_name="$project_name" \
    --data_flag="$data_flag" \
    --arch="$arch" \
    --block_type="$block_type" \
    --pool_method="$pool_method" \
    --final_pool_method="$final_pool" \
    --target_resolution="$target_res" \
    --batch_size=32 \
    --num_epochs=100 \
    --num_workers=4 \
    --scheduler="WarmupCosineAnnealingLR" \
    --max_lr=1e-4 \
    --weight_decay=0.05 \
    --warmup_epochs=10 \
    --stop_patience=20 \
    --download \
    --output_root="/scratch/work/yuy10/DINOv3/output_dir" \
    --seed="$target_seed" \
    --training_method="FT" \
    --dataset_type=Hybrid_2D3D \
    --ratio_2d=1 \
    --ratio_3d=1 \
    --lambda_2d=1.0 \
    --lambda_3d=1.0 \
    --updates_per_epoch=37 \
    --dataset2D=chestmnist \
    --dataset3D=nodulemnist3d \
    --batchsize2d=128 \
    --batchsize3d=32 \
    $extra_args   \
    --cycle_order $target_cycle  # 注意：这里不加引号，Bash 会将其拆分为 HW DH DW

echo "===== JOB END ====="
date
