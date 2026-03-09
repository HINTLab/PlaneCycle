# SLURM Job Submission Guide

This repository contains scripts for training DINOv3 models on MedMNIST datasets using SLURM job scheduling.

## Quick Start

### 1. Configure Environment Variables

Edit `submit_job.sh` and set the following variables to match your system:

```bash
ENV_NAME=""                    # Conda environment name
PROJECT_ROOT="/path/to/project"      # Project root directory
WEIGHT_DIR="/path/to/weights"        # Directory for pretrained weights
OUTPUT_DIR="/path/to/outputs"        # Directory for training outputs
ENTITY="your-wandb-entity"           # Weights & Biases entity name
TRAIN_SCRIPT="${PROJECT_ROOT}/experiments/medmnist/train_eval.py"  # Path to training script
```

### 2. Configure Training Hyperparameters (Optional)

Modify these variables in `submit_job.sh` if needed:

```bash
BATCH_SIZE=32                        # Batch size for training
NUM_EPOCHS=200                       # Number of training epochs
NUM_WORKERS=4                        # Number of DataLoader workers
SCHEDULER="WarmupCosineAnnealingLR" # Learning rate scheduler
MAX_LR=1e-3                          # Maximum learning rate
WEIGHT_DECAY=1e-5                    # Weight decay for optimizer
WARMUP_EPOCHS=10                     # Warmup epochs
TRAINING_METHOD="LP"                 # Training method (LP for linear probing)
DOWNLOAD_FLAG="--download"           # Use --download to auto-download datasets
```

### 3. Configure Search Space (Optional)

Modify arrays in `submit_job.sh` to control which experiments to run:

```bash
seeds=(42 123 1024 1337 2026)
datasets=(
  "nodulemnist3d"
  "organmnist3d"
  "adrenalmnist3d"
  ...
)
arch_list=("dinov3_vitl16" "dinov3_vitb16" "dinov3_vits16")
configs=("PlaneCycle:PCg" "PlaneCycle:PCm" "Slice2D:" "Flatten3D:")
cycle_orders=("HW DW DH HW")
```

### 4. Submit Jobs

```bash
sbatch submit_job.sh
```

This will submit a job array. The number of jobs equals:
```
total_jobs = len(seeds) × len(datasets) × len(arch_list) × len(configs) × len(cycle_orders)
```


## Directory Structure

Ensure your project has the following structure:

```
PROJECT_ROOT/
├── experiments/
│   └── medmnist/
│       └── train_eval.py          # Training script
├── models/                        # Contains model definitions (for torch.hub)
└── logs/                          # SLURM logs (created automatically)

WEIGHT_DIR/                        # Pretrained weights directory
OUTPUT_DIR/                        # Training outputs and checkpoints
```

## Configuration Tips

### To run a single experiment:
Set all arrays to have single elements:
```bash
seeds=(42)
datasets=("organmnist3d")
arch_list=("dinov3_vits16")
```

### To modify SLURM resources:
Edit the SBATCH header in `submit_job.sh`:
```bash
#SBATCH --partition=gpu-h200-141g-ellis,gpu-h200-141g-short
#SBATCH --gpus=1                    # Number of GPUs
#SBATCH --cpus-per-task=4          # CPUs per task
#SBATCH --mem=128G                 # Memory per task
#SBATCH -t 8:00:00                 # Time limit
```

## Notes
- Each job runs a single experiment configuration
- Jobs are independent and can run in parallel
- Results are logged to Weights & Biases (W&B)
- Training outputs are saved to `OUTPUT_DIR`
- Logs are saved to `logs/` directory with naming pattern `job_JOBID_ARRAYID.out`
