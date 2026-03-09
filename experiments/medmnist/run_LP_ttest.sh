#!/bin/bash
#SBATCH --partition=gpu-h200-141g-ellis,gpu-h200-141g-short
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --array=0-23   # 大范围占位，脚本内部自动检查
#SBATCH -t 8:00:00
#SBATCH -J dinov3_LP_ttest
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
seeds=(42
#      123
#      1024
#      1337
#      2026
      )


datasets=(
  "nodulemnist3d"
  "organmnist3d"
  "adrenalmnist3d"
  "fracturemnist3d"
  "vesselmnist3d"
  "synapsemnist3d"
)

arch_list=(
#  "dinov3_vitl16"
#  "dinov3_vitb16"
  "dinov3_vits16"
)

final_pool_methods=("learn_to_pool")

configs=(
  "PlaneCycle:PCg"
  "PlaneCycle:PCm"
  "Slice2D:"
  "Flatten3D:"
)

cycle_orders=(
#  "HW DW DH"
  "HW DW DH HW"
)

resolutions=(64)

name="reproduce"

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
echo "Cycle Order    : $target_cycle"
echo "======================"

# --- 运行 ---
python experiments/medmnist/train_eval.py \
    --weight_dir="/scratch/work/yuy10/DINOv3/weight_dir" \
    --entity="yuyinghong1-aalto-university" \
    --project_name="$project_name" \
    --data_flag="$data_flag" \
    --arch="$arch" \
    --block_type="$block_type" \
    --pool_method="$pool_method" \
    --final_pool_method="$final_pool" \
    --batch_size=32 \
    --num_epochs=200 \
    --num_workers=4 \
    --scheduler="WarmupCosineAnnealingLR" \
    --max_lr=1e-3 \
    --weight_decay=1e-5 \
    --warmup_epochs=10 \
    --download \
    --output_root="/scratch/work/yuy10/DINOv3/output_dir" \
    --seed="$target_seed" \
    --training_method="LP" \
    --cycle_order $target_cycle

echo "===== JOB END ====="
date
