#!/bin/bash
#SBATCH --partition=gpu-h200-141g-ellis,gpu-h200-141g-short
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --array=0-360%20
#SBATCH -t 8:00:00
#SBATCH -J dinov3_LP_training
#SBATCH -o logs/job_%A_%a.out

mkdir -p logs
set -euo pipefail

echo "===== JOB START ====="
date
hostname

# Environment setup
if [ -f "$WRKDIR/.bashrc.sh" ]; then
  source "$WRKDIR/.bashrc.sh"
elif [ -f ".bashrc.sh" ]; then
  source ".bashrc.sh"
fi

module --force purge
module load mamba

# Configuration variables
ENV_NAME=""
PROJECT_ROOT=""
WEIGHT_DIR=""
OUTPUT_DIR=""
ENTITY=""
TRAIN_SCRIPT="${PROJECT_ROOT}/experiments/medmnist/train_eval.py"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"
source activate "${ENV_NAME}"

# Training hyperparameters
BATCH_SIZE=32
NUM_EPOCHS=200
NUM_WORKERS=4
SCHEDULER="WarmupCosineAnnealingLR"
MAX_LR=1e-3
WEIGHT_DECAY=1e-5
WARMUP_EPOCHS=10
TRAINING_METHOD="LP"
DOWNLOAD_FLAG="--download"

# Search space
seeds=(42
      123
      1024
      1337
      2026
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
  "dinov3_vitl16"
  "dinov3_vitb16"
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
  "HW DW DH HW"
)

resolutions=(64)

experiment_name="reproduce"

# Calculate total number of jobs
total=$(( ${#seeds[@]} * ${#datasets[@]} * ${#arch_list[@]} * ${#final_pool_methods[@]} * ${#configs[@]} * ${#resolutions[@]} * ${#cycle_orders[@]} ))

echo "Total jobs: $total"
echo "Current SLURM task ID: ${SLURM_ARRAY_TASK_ID:-0}"

if [[ ${SLURM_ARRAY_TASK_ID:-0} -ge $total ]]; then
  echo "Task ID exceeds total jobs — exiting."
  exit 0
fi

task_id=${SLURM_ARRAY_TASK_ID:-0}

# Decompose task index
n_seed=${#seeds[@]}
n_data=${#datasets[@]}
n_arch=${#arch_list[@]}
n_fpool=${#final_pool_methods[@]}
n_conf=${#configs[@]}
n_res=${#resolutions[@]}
n_cycle=${#cycle_orders[@]}

cycle_idx=$(( task_id % n_cycle )); task_id=$(( task_id / n_cycle ))
res_idx=$(( task_id % n_res )); task_id=$(( task_id / n_res ))
conf_idx=$(( task_id % n_conf )); task_id=$(( task_id / n_conf ))
f_idx=$(( task_id % n_fpool )); task_id=$(( task_id / n_fpool ))
arch_idx=$(( task_id % n_arch )); task_id=$(( task_id / n_arch ))
data_idx=$(( task_id % n_data )); task_id=$(( task_id / n_data ))
seed_idx=$(( task_id % n_seed ))

target_cycle=${cycle_orders[$cycle_idx]}
target_seed=${seeds[$seed_idx]}
target_res=${resolutions[$res_idx]}
arch=${arch_list[$arch_idx]}
final_pool=${final_pool_methods[$f_idx]}
data_flag=${datasets[$data_idx]}
project_name="${arch}_${experiment_name}_${target_seed}"

current_config=${configs[$conf_idx]}
block_type=${current_config%:*}
pool_method=${current_config#*:}

# Print configuration
echo "===== RUN CONFIG ====="
echo "Dataset        : $data_flag"
echo "Architecture   : $arch"
echo "Config         : $current_config"
echo "Block type     : $block_type"
echo "Pool method    : $pool_method"
echo "Final pool     : $final_pool"
echo "Resolution     : $target_res"
echo "Seed           : $target_seed"
echo "Cycle order    : $target_cycle"
echo "Batch size     : $BATCH_SIZE"
echo "Num epochs     : $NUM_EPOCHS"
echo "Learning rate  : $MAX_LR"
echo "======================"

python "$TRAIN_SCRIPT" \
    --weight_dir="$WEIGHT_DIR" \
    --entity="$ENTITY" \
    --project_name="$project_name" \
    --data_flag="$data_flag" \
    --arch="$arch" \
    --block_type="$block_type" \
    --pool_method="$pool_method" \
    --final_pool_method="$final_pool" \
    --batch_size="$BATCH_SIZE" \
    --num_epochs="$NUM_EPOCHS" \
    --num_workers="$NUM_WORKERS" \
    --scheduler="$SCHEDULER" \
    --max_lr="$MAX_LR" \
    --weight_decay="$WEIGHT_DECAY" \
    --warmup_epochs="$WARMUP_EPOCHS" \
    $DOWNLOAD_FLAG \
    --output_root="$OUTPUT_DIR" \
    --seed="$target_seed" \
    --training_method="$TRAINING_METHOD" \
    --cycle_order $target_cycle

echo "===== JOB END ====="
date