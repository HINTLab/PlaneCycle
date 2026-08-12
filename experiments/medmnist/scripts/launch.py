#!/usr/bin/env python
"""Slurm-array launcher driven by YAML configs.

One array task = one grid point, running all datasets sequentially.

- dinov3 family: grid is model_jobs (arch x block_type x pool x cycle)
  x resolutions x final_pool x seeds.
- spectre / ctfm families: a fixed single backbone, so the grid is just seeds.

Usage (--config takes a short name under configs/, e.g. planecycle/lp):
    python launch.py --config planecycle/lp --count           # array size
    python launch.py --config planecycle/lp --task-id 3       # run task 3
    python launch.py --config planecycle/lp --task-id 3 --dry-run

Instead of a numeric task-id you can pin a specific setting with overrides
(--arch / --block-type / --pool / --cycle-order / --seed / --family); fully
specified they narrow the grid to one task, so just run it:
    python launch.py --config planecycle/convnext_lp --cycle-order "HW DW DH" --seed 42
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def resolve_config(value):
    """Accept a short sweep name ('planecycle/lp') or an explicit path.
    The short form resolves to configs/<value>.yaml next to this script."""
    if os.path.isfile(value):
        return value
    here = os.path.dirname(os.path.abspath(__file__))
    name = value if value.endswith(".yaml") else f"{value}.yaml"
    return os.path.join(here, "configs", name)


def load_config(path):
    """Load a sweep YAML and merge in the shared private paths from
    scripts/paths.yaml (entity / weight_dir / output_root / spectre_weight_path).
    The sweep config stays self-contained for everything else; a sweep may
    still override a path by setting the key itself."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    here = os.path.dirname(os.path.abspath(__file__))
    paths_file = os.path.join(here, "configs", "paths.yaml")
    if not os.path.isfile(paths_file):
        raise SystemExit(
            f"Missing {paths_file}\n"
            "Copy the template and fill in your paths:\n"
            "    cp configs/paths.yaml.example configs/paths.yaml"
        )
    with open(paths_file) as f:
        for key, value in (yaml.safe_load(f) or {}).items():
            cfg.setdefault(key, value)
    return cfg


def build_model_jobs(cfg):
    """(arch, block_type, pool_method, cycle_order) combinations.

    - dinov3: PlaneCycle expands pool_methods x cycle_orders (ConvNeXt has no
      global tokens -> no pool_method); other block types need neither.
    - spectre / ctfm: a single fixed backbone -> one degenerate job so the
      only remaining sweep axis is the seed.
    """
    if cfg.get("model_family", "dinov3") != "dinov3":
        return [(None, None, None, None)]
    jobs = []
    for arch in cfg["archs"]:
        for block_type in cfg["block_types"]:
            if block_type == "PlaneCycle":
                for cycle in cfg["cycle_orders"]:
                    if "convnext" in arch:
                        jobs.append((arch, block_type, None, cycle))
                    else:
                        for pool_method in cfg["pool_methods"]:
                            jobs.append((arch, block_type, pool_method, cycle))
            else:
                jobs.append((arch, block_type, None, None))
    return jobs


def decode_task(cfg, task_id):
    """Map a flat array index to one grid point (model axis fastest,
    matching the old bash decode order). Axes absent from the config
    (resolutions / final_pool for spectre/ctfm) collapse to a single point."""
    axes = [
        ("model", build_model_jobs(cfg)),
        ("resolution", cfg.get("resolutions", [None])),
        ("final_pool", cfg.get("final_pool_methods", [None])),
        ("seed", cfg["seeds"]),
    ]
    total = 1
    for _, values in axes:
        total *= len(values)
    if task_id >= total:
        return None, total
    picked, i = {}, task_id
    for name, values in axes:
        picked[name] = values[i % len(values)]
        i //= len(values)
    return picked, total


def _train_eval_path():
    """train_eval.py, resolved via $ROOT_DIR or this file's location."""
    default_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    return os.path.join(
        os.environ.get("ROOT_DIR", default_root), "experiments/medmnist/train_eval.py"
    )


def project_name(cfg, arch, seed):
    """W&B project (the grouping bucket) — derived from fields, no name in the
    YAML, so launch and analyze always agree. dinov3 keys on arch + method +
    seed (so every method on the same backbone lands in ONE project and its
    curves can be compared directly, distinguished by block_type / cycle_order
    in each run's config); spectre/ctfm key on the family. Must NOT contain a
    timestamp: analyze_wandb.py reconstructs this string to fetch results."""
    family = cfg.get("model_family", "dinov3")
    method = cfg["train_args"]["training_method"]
    if family == "dinov3":
        return f"{arch}_{method}_{seed}"
    return f"{family}_{method}_{seed}"


def run_name(cfg, picked, data_flag, stamp):
    """W&B run name (one run = one dataset). Arch and seed are already in the
    project name, so they're dropped here; a timestamp makes re-launches
    distinguishable. Purely cosmetic — analyze filters on run config, not this."""
    family = cfg.get("model_family", "dinov3")
    if family == "dinov3":
        _, block_type, pool_method, _ = picked["model"]
        parts = [data_flag, block_type] + ([pool_method] if pool_method else [])
    else:
        parts = [data_flag, family]
    return "_".join(parts + [stamp])


def build_command(cfg, picked, data_flag, stamp=""):
    family = cfg.get("model_family", "dinov3")
    seed = picked["seed"]
    arch = picked["model"][0]
    project = project_name(cfg, arch, seed)

    cmd = [
        sys.executable,
        _train_eval_path(),
        f"--entity={cfg['entity']}",
        f"--project_name={project}",
        f"--run_name={run_name(cfg, picked, data_flag, stamp)}",
        f"--data_flag={data_flag}",
        f"--output_root={cfg['output_root']}",
        f"--seed={seed}",
        "--download",
    ]
    for key, value in cfg["train_args"].items():
        cmd.append(f"--{key}={value}")

    if family == "dinov3":
        _, block_type, pool_method, cycle_order = picked["model"]
        cmd += [
            f"--weight_dir={cfg['weight_dir']}",
            f"--arch={arch}",
            f"--block_type={block_type}",
            f"--final_pool_method={picked['final_pool']}",
            f"--target_resolution={picked['resolution']}",
        ]
        if block_type == "PlaneCycle":
            if pool_method:
                cmd += ["--pool_method", pool_method]
            cmd += ["--cycle_order", *cycle_order.split()]
    elif family == "spectre":
        cmd += ["--model_family=spectre",
                f"--spectre_weight_path={cfg['spectre_weight_path']}"]
    elif family == "ctfm":
        cmd += ["--model_family=ctfm"]
    else:
        raise ValueError(f"Unknown model_family: {family}")
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="a sweep YAML")
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Slurm array task id (defaults to $SLURM_ARRAY_TASK_ID)",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Print the total number of array tasks and exit",
    )
    parser.add_argument(
        "--array",
        action="store_true",
        help="Print the Slurm array range '0-(N-1)' and exit "
        "(feed straight to sbatch --array)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the commands without running them"
    )
    # overrides: narrow the grid to a specific setting (no task-id guessing).
    parser.add_argument("--arch", help="run only this arch")
    parser.add_argument("--block-type", help="run only this block_type")
    parser.add_argument("--pool", help="run only this pool_method (PlaneCycle)")
    parser.add_argument("--cycle-order", help='run only this cycle_order, e.g. "HW DW DH"')
    parser.add_argument("--seed", type=int, help="run only this seed")
    parser.add_argument("--family", help="override model_family (spectre/ctfm)")
    args = parser.parse_args()

    cfg_path = resolve_config(args.config)
    cfg = load_config(cfg_path)
    cfg["group"] = Path(cfg_path).parent.name  # planecycle / baselines / fm3d

    # apply overrides — each pins one axis, shrinking the grid (fully specified
    # -> a single task, so no --task-id needed)
    if args.family:
        cfg["model_family"] = args.family
    if args.arch:
        cfg["archs"] = [args.arch]
    if args.block_type:
        cfg["block_types"] = [args.block_type]
    if args.pool:
        cfg["pool_methods"] = [args.pool]
    if args.cycle_order:
        cfg["cycle_orders"] = [args.cycle_order]
    if args.seed is not None:
        cfg["seeds"] = [args.seed]

    task_id = args.task_id
    if task_id is None:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    picked, total = decode_task(cfg, task_id)

    if args.count:
        print(total)
        return
    if args.array:
        print(f"0-{total - 1}")
        return
    if picked is None:
        print(f"Task id {task_id} >= total {total} - nothing to do.")
        return

    family = cfg.get("model_family", "dinov3")
    arch, block_type, pool_method, cycle_order = picked["model"]
    print("===== TASK CONFIG =====")
    print(f"Task ID     : {task_id} / {total - 1}")
    print(f"Family      : {family}")
    if family == "dinov3":
        print(f"Arch        : {arch}")
        print(f"Block type  : {block_type}")
        print(f"Pool method : {pool_method or 'N/A'}")
        print(f"Cycle order : {cycle_order or 'N/A'}")
        print(f"Final pool  : {picked['final_pool']}")
        print(f"Resolution  : {picked['resolution']}")
    print(f"Seed        : {picked['seed']}")
    print(f"Datasets    : {' '.join(cfg['datasets'])}")
    print("=======================")

    stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    for data_flag in cfg["datasets"]:
        cmd = build_command(cfg, picked, data_flag, stamp)
        print(
            f"\n===== {'DRY-RUN' if args.dry_run else 'RUN'} DATASET: {data_flag} ====="
        )
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
