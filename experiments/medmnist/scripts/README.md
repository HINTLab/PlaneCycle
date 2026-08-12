# Sweep launcher (`scripts/`)

Tooling to launch the MedMNIST sweeps on Slurm and pull the results back from
Weights & Biases. Your machine-specific settings live in three git-ignored files
(`configs/paths.yaml`, `configs/env.sh`, `run.sh`); everything else is portable.
Slurm is optional — `launch.py` can also run any grid point directly on a GPU
machine.

Each sweep is one **self-contained YAML** under `configs/`. There is no shared
base or inheritance: what a sweep runs is fully described by its own file.

## Setup

```bash
cd experiments/medmnist/scripts
cp configs/paths.yaml.example configs/paths.yaml   # then edit: W&B entity, weight_dir, output_root, ...
cp configs/env.sh.example    configs/env.sh        # then edit: interpreter, Slurm partition, conda env
cp run.sh.example            run.sh                # then edit: repo root, conda env, #SBATCH partition
```

All three are git-ignored, so your paths, partition names and environment never
get committed. The `*.example` templates are committed as documentation — each
one lists every field with a comment, so the copy is the only file you edit.

`CONDA_ENV` and the partition appear in both `env.sh` and `run.sh`: `submit.sh`
reads them from `env.sh` and passes `--partition` on the command line, which
overrides the `#SBATCH` line in `run.sh`. Setting them in `run.sh` too keeps a
bare `sbatch run.sh ...` (without `submit.sh`) working.

## Files

| File | Role |
|------|------|
| `configs/<group>/<lp\|ft>.yaml` | One sweep definition (axes + `train_eval.py` args). |
| `configs/paths.yaml` | Private paths + W&B entity (`entity`, `weight_dir`, `output_root`, `spectre_weight_path`), merged into every config by `launch.py`. Git-ignored; copy from `configs/paths.yaml.example`. |
| `configs/env.sh` | Private cluster settings for the shell scripts (`PY`, `PARTITION`, `CONDA_ENV`). Git-ignored; copy from `configs/env.sh.example`. |
| `launch.py` | Reads a config, expands the sweep grid, maps one Slurm array index → one run set, and calls `train_eval.py`. |
| `run.sh` | Slurm array job script: activates the env and runs `launch.py` for `$SLURM_ARRAY_TASK_ID`. Set `ROOT_DIR`, `CONDA_ENV` and the `#SBATCH --partition` line inside it. Git-ignored; copy from `run.sh.example`. |
| `submit.sh` | Convenience wrapper: computes the array size, then `sbatch run.sh` with your partition. No self-submission. |
| `analyze_wandb.py` | Reads the same config, fetches finished runs from W&B, prints per-seed + mean tables, writes CSVs. |

## Config layout

```
configs/
  planecycle/{lp,ft}.yaml           # the paper's method (PlaneCycle); has pool/cycle
  planecycle/convnext_{lp,ft}.yaml  # same, on the ConvNeXt backbones
  baselines/{lp,ft}.yaml            # DINOv3 lifting baselines: Slice2D (2D) / Flatten3D (3D) / TriSlice
  baselines/convnext_{lp,ft}.yaml   # same, on ConvNeXt (+ convnext_acs_{lp,ft}.yaml for ACS)
  fm3d/{lp,ft}.yaml                 # natively-3D foundation models: SPECTRE / CT-FM (pick one via model_family)
```

- **Sweep-focused**: each config lists what to run (seeds, datasets, axes,
  `train_args`); the private paths (`entity`, `weight_dir`, `output_root`,
  `spectre_weight_path`) live once in `paths.yaml` and are merged in at load
  time. Copy a config file to make a new sweep; edit `paths.yaml` to relocate.
- **Enable methods by (un)commenting** the `block_types` list. In
  `baselines/*.yaml` only `Slice2D` is on by default; uncomment `Flatten3D` /
  `TriSlice` to sweep them too.
- **fm3d picks a backbone** with `model_family: spectre` / `ctfm` (comment the
  other). `spectre_weight_path` is used only for SPECTRE.
- **No blank lines / no `name` field**: names are derived (see below).

## How the grid maps to array tasks

`build_model_jobs()`:
- dinov3: expands `archs × block_types`; `PlaneCycle` additionally fans out
  over `pool_methods × cycle_orders` (ConvNeXt has no global tokens → no
  `pool_method`); other block types take neither.
- fm3d (spectre/ctfm): one fixed backbone, so a single degenerate job.

`decode_task()` indexes `model_jobs × resolutions × final_pool × seeds` (fm3d
collapses the middle axes, leaving just seeds). **One array task = one grid
point, and it runs all `datasets` sequentially.**

## Running a sweep

### Check the sweep before submitting

`launch.py` works without Slurm, so you can inspect a config and test one
grid point first (on a GPU machine, or inside an interactive `srun` shell):

```bash
cd experiments/medmnist/scripts

python launch.py --config planecycle/lp --count                # grid size
python launch.py --config planecycle/lp --task-id 0 --dry-run  # print the exact commands

# quick real run: temporarily trim the YAML (one dataset, a few epochs), then
WANDB_MODE=offline python launch.py --config planecycle/lp --seed 42
```

`WANDB_MODE=offline` keeps test runs out of your real W&B projects.

### Submit the full array

Once a task runs cleanly, submit from the cluster login node:

```bash
cd experiments/medmnist/scripts

./submit.sh planecycle/lp      # PlaneCycle, Linear Probing
./submit.sh baselines/ft       # 2D/3D/TriSlice baselines, Fine-Tuning
./submit.sh fm3d/lp            # SPECTRE / CT-FM (family chosen in the yaml)
```

`submit.sh` computes the array range from the config (so it always matches the
sweep) and calls `sbatch run.sh <group>/<mode>`.

### Reproduce one number instead of the whole grid

A full sweep is hundreds of GPU-hours. To check a single cell of a results
table, append override flags after the config: the grid — and the array size —
narrows to match, so you never have to guess a task-id.

`--arch` · `--block-type` · `--pool` · `--cycle-order` · `--seed` · `--family`

```bash
# submit just one setting via Slurm
./submit.sh planecycle/convnext_lp --cycle-order "HW DW DH" --seed 42
./submit.sh baselines/convnext_lp  --block-type ACS --seed 42
```

The same overrides work on `launch.py` to **run locally, no Slurm** (do it on a
GPU node / inside `srun`, with the env active):

```bash
# one PlaneCycle setting (3-cycle, seed 42) — all 6 datasets, real W&B
python launch.py --config planecycle/convnext_lp --cycle-order "HW DW DH" --seed 42

# a specific ViT PlaneCycle point
python launch.py --config planecycle/lp --arch dinov3_vitb16 --pool PCg --seed 123

# force a block type onto a config (e.g. ACS / TriSlice on ConvNeXt/ViT)
python launch.py --config baselines/convnext_lp --block-type ACS --seed 42

# check what a narrowed selection resolves to before running
python launch.py --config planecycle/convnext_lp --block-type PlaneCycle --seed 42 --count
python launch.py --config planecycle/convnext_lp --cycle-order "HW DW DH" --seed 42 --dry-run

# don't touch the real W&B projects (offline)
WANDB_MODE=offline python launch.py --config baselines/convnext_lp --block-type ACS --seed 42
```

Fully specified, the overrides leave a single task, so it just runs; otherwise
it runs task 0 of the narrowed grid (add `--task-id N` to pick another). (To
rerun tasks by index instead, use the raw form:
`sbatch --array=3,7,12 run.sh planecycle/lp`.)

Prefer plain `sbatch`? Do the two steps yourself (`--array` prints the
`0-(N-1)` range for the config):

```bash
PY=python
sbatch --array=$($PY launch.py --config planecycle/lp --array) \
       run.sh planecycle/lp
```

`launch.py` finds `train_eval.py` via `$ROOT_DIR` (set in `run.sh`), falling
back to the repo root inferred from its own path — so it works both under
Slurm and when run directly.

## W&B naming (project vs run)

Derived from config fields — never written in the YAML — so `launch.py` and
`analyze_wandb.py` always agree.

| | Pattern | Example |
|---|---|---|
| **project** (grouping bucket) | dinov3: `{arch}_{method}_{seed}` · fm3d: `{family}_{method}_{seed}` | `dinov3_vits16_LP_42`, `spectre_LP_42` |
| **run** (one dataset) | dinov3: `{data}_{block}_{pool}_{stamp}` · fm3d: `{data}_{family}_{stamp}` | `nodulemnist3d_PlaneCycle_PCg_260714_1514` |

- **Every method on the same backbone shares one project** (`planecycle`,
  `baselines`, ... all land in `{arch}_{method}_{seed}`), distinguished by
  `block_type` / `cycle_order` in each run's config — so their curves can be
  compared directly in one W&B project.
- The project name must stay reconstructable, so it carries **no timestamp**
  (`analyze_wandb.py` rebuilds it to fetch results). The timestamp lives in the
  **run** name, where re-launches need to be distinguishable.

## Collecting results

```bash
python analyze_wandb.py --config planecycle/lp     # -> ./wandb_results/LP/*.csv
python analyze_wandb.py --config fm3d/ft --output_dir ./results
python analyze_wandb.py --config planecycle/lp --all-methods   # every method in the project, one table
```

It reuses `launch.build_model_jobs()` and `launch.project_name()`, so the
analysis grid and project names always match what was launched. Only
`state == "finished"` runs are counted. Since every method on a backbone now
shares one project, **`--all-methods`** builds the row list from the runs
actually present (all `block_type` / `cycle_order` combos) instead of just the
config's — one table comparing PlaneCycle / Slice2D / ACS / ... directly.
