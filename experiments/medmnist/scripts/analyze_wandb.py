"""
WandB Results Analyzer
======================
Reads the same YAML config as launch.py and prints one table per seed plus a
mean-across-seeds table per architecture. Family, LP/FT and the model-job
expansion all come from the config.

Usage:
    python analyze_wandb.py --config planecycle/lp
    python analyze_wandb.py --config fm3d/ft --output_dir ./results
    python analyze_wandb.py --config planecycle/lp --all-methods  # all methods in
                                                                  # the shared project
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

import launch  # config loading + model-job expansion shared with the launcher

warnings.filterwarnings("ignore")

DATASET_SHORT = {
    "nodulemnist3d": "Nodule",
    "organmnist3d": "Organ",
    "adrenalmnist3d": "Adrenal",
    "fracturemnist3d": "Fracture",
    "vesselmnist3d": "Vessel",
    "synapsemnist3d": "Synapse",
}

AUC_KEY = "final_test_auc"
ACC_KEY = "final_test_acc"

DATASET_PRIORITY = [
    "organmnist3d",
    "nodulemnist3d",
    "fracturemnist3d",
    "adrenalmnist3d",
    "vesselmnist3d",
    "synapsemnist3d",
]


def _ordered_datasets(datasets: list[str]) -> list[str]:
    dataset_set = set(datasets)
    return [ds for ds in DATASET_PRIORITY if ds in dataset_set]


# ─────────────────────────────────────────────────────────────
# Model jobs (expansion shared with launch.py, labels added here)
# ─────────────────────────────────────────────────────────────


FAMILY_LABEL = {"spectre": "SPECTRE", "ctfm": "CT-FM"}


def _row_label(family, block_type, pool_method, cycle_order):
    if family != "dinov3":
        return FAMILY_LABEL.get(family, family)
    if block_type != "PlaneCycle":
        return block_type
    if pool_method:
        return f"PlaneCycle-{pool_method} [{cycle_order}]"
    return f"PlaneCycle [{cycle_order}]"


def _job(family, arch, block_type, pool_method, cycle_order):
    return dict(
        arch=arch,
        block_type=block_type,
        pool_method=pool_method or "",
        cycle_order=cycle_order,
        row_label=_row_label(family, block_type, pool_method, cycle_order),
    )


def build_labeled_jobs(cfg: dict) -> list[dict]:
    """One row per model job the config sweeps."""
    family = cfg.get("model_family", "dinov3")
    jobs = [_job(family, *mj) for mj in launch.build_model_jobs(cfg)]
    return _paper_order(jobs)


def _paper_order(jobs: list[dict]) -> list[dict]:
    """Order rows like the paper tables.

    Baselines precede PlaneCycle. Within PlaneCycle, PCm precedes PCg (the
    paper's row order), and shorter ConvNeXt cycles precede the four-cycle.
    """
    def key(job):
        block = job["block_type"] or ""
        if block == "PlaneCycle":
            block_rank = 2
            pool_rank = {"PCm": 0, "PCg": 1}.get(job["pool_method"], 2)
            cycle = job["cycle_order"] or ""
            cycle_rank = len(cycle.split()) if cycle else 0
            return (block_rank, pool_rank, cycle_rank, cycle)
        # Paper-style baseline order for the ConvNeXt comparison.
        baseline_rank = {"Slice2D": 0, "ACS": 1}.get(block, 0)
        return (baseline_rank, 0, 0, block)

    return sorted(jobs, key=key)


def jobs_from_runs(df, family) -> list[dict]:
    """One row per (block_type, pool_method, cycle_order) actually present in
    the fetched runs — used by --all-methods so a shared project shows every
    method (PlaneCycle cycle orders, Slice2D, ACS, ...) in one table."""
    combos = (
        df[["block_type", "pool_method", "cycle_order"]]
        .fillna("")
        .drop_duplicates()
        .sort_values(["block_type", "pool_method", "cycle_order"])
    )
    jobs = [
        _job(family, None, r["block_type"], r["pool_method"], r["cycle_order"])
        for _, r in combos.iterrows()
    ]
    return _paper_order(jobs)


# ─────────────────────────────────────────────────────────────
# WandB fetching
# ─────────────────────────────────────────────────────────────


def _normalise_cycle(value) -> str:
    if value is None:
        return ""
    return " ".join(str(v) for v in value) if isinstance(value, list) else str(value).strip()


def fetch_runs(arch, cfg: dict, api: wandb.Api) -> pd.DataFrame:
    rows = []
    for seed in cfg["seeds"]:
        proj = launch.project_name(cfg, arch, seed)  # same string launch.py wrote
        try:
            runs = api.runs(f"{cfg['entity']}/{proj}", per_page=500)
            n = 0
            for run in runs:
                if run.state != "finished":
                    continue
                c, s = run.config, run.summary._json_dict
                auc, acc = s.get(AUC_KEY), s.get(ACC_KEY)
                if auc is None and acc is None:
                    continue
                rows.append(
                    dict(
                        seed=c.get("seed", seed),
                        dataset=c.get("dataset"),
                        block_type=c.get("block_type"),
                        pool_method=c.get("pool_method") or "",
                        cycle_order=_normalise_cycle(c.get("cycle_order")),
                        final_test_auc=auc,
                        final_test_acc=acc,
                    )
                )
                n += 1
            print(f"  [{proj}] {n} runs with metrics")
        except Exception as exc:
            print(f"  WARNING: {proj} — {exc}")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# Table building
# ─────────────────────────────────────────────────────────────


def _match_job(df: pd.DataFrame, job: dict) -> pd.DataFrame:
    if job["block_type"] is None:  # spectre/ctfm: single group, no filtering
        return df
    mask = df["block_type"] == job["block_type"]
    if job["block_type"] == "PlaneCycle":
        mask &= df["pool_method"] == job["pool_method"]
        if job["cycle_order"]:
            mask &= df["cycle_order"] == job["cycle_order"]
    return df[mask]


def _make_columns(datasets: list[str]) -> pd.MultiIndex:
    col_tuples = [(DATASET_SHORT.get(ds, ds), m) for ds in datasets for m in ("AUC", "ACC")]
    col_tuples += [("Average", "AUC"), ("Average", "ACC")]
    return pd.MultiIndex.from_tuples(col_tuples)


def build_table(df: pd.DataFrame, arch_jobs: list[dict], datasets: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["pool_method"] = df["pool_method"].fillna("")
    df["cycle_order"] = df["cycle_order"].fillna("")

    result_rows = []
    for job in arch_jobs:
        sub = _match_job(df, job)
        row_vals, auc_means, acc_means = [], [], []
        for ds in datasets:
            ds_sub = sub[sub["dataset"] == ds]
            aucs = ds_sub["final_test_auc"].dropna() if not ds_sub.empty else pd.Series(dtype=float)
            accs = ds_sub["final_test_acc"].dropna() if not ds_sub.empty else pd.Series(dtype=float)
            row_vals.append(f"{aucs.mean():.4f}" if len(aucs) else "—")
            if len(aucs):
                auc_means.append(aucs.mean())
            row_vals.append(f"{accs.mean():.4f}" if len(accs) else "—")
            if len(accs):
                acc_means.append(accs.mean())
        row_vals.append(f"{np.mean(auc_means):.4f}" if auc_means else "—")
        row_vals.append(f"{np.mean(acc_means):.4f}" if acc_means else "—")
        result_rows.append((job["row_label"], row_vals))

    return pd.DataFrame(
        [r[1] for r in result_rows],
        index=[r[0] for r in result_rows],
        columns=_make_columns(datasets),
    )


# ─────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────


def print_table(table: pd.DataFrame, title: str, datasets: list[str]):
    width = 110
    print(f"\n{'='*width}\n  {title}\n{'='*width}")
    if table.empty:
        print("  (no data)")
        return
    ds_names = [t[0] for t in table.columns]
    prev, h1 = None, []
    for ds in ds_names:
        h1.append(f"{ds:>18}" if ds != prev else " " * 18)
        prev = ds
    print(f"{'Setting':<40}" + "".join(h1))
    h2 = [f"{'AUC':>9}{'ACC':>9}" for _ in datasets] + [f"{'Avg AUC':>9}{'Avg ACC':>9}"]
    print(f"{'':40}" + "".join(h2))
    print("-" * width)
    for idx, row in table.iterrows():
        print(f"{str(idx):<40}" + "".join(f"{v:>9}" for v in row.values))
    print("=" * width)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="short sweep name under configs/, e.g. planecycle/lp")
    p.add_argument("--save_csv", action="store_true", default=True)
    p.add_argument("--output_dir", default=None,
                   help="Default: ./wandb_results/<LP|FT>")
    p.add_argument("--all-methods", action="store_true",
                   help="Show every method found in the project (not just this "
                   "config's) — one table comparing PlaneCycle / Slice2D / ACS / ...")
    return p.parse_args()


def main():
    args = parse_args()
    cfg_path = launch.resolve_config(args.config)
    print(f"Reading config from: {cfg_path}")
    cfg = launch.load_config(cfg_path)
    cfg["group"] = Path(cfg_path).parent.name  # matches launch.py project naming
    cfg["datasets"] = _ordered_datasets(cfg["datasets"])
    mode = cfg["train_args"].get("training_method", "LP")
    output_dir = Path(args.output_dir or f"./wandb_results/{mode}")

    family = cfg.get("model_family", "dinov3")
    # dinov3 groups results by architecture; spectre/ctfm have one fixed
    # backbone, so there is a single group and no arch prefix.
    groups = cfg["archs"] if family == "dinov3" else [None]

    print(f"  entity        : {cfg['entity']}")
    print(f"  family        : {family}")
    print(f"  mode          : {mode}")
    print(f"  seeds         : {cfg['seeds']}")
    print(f"  datasets      : {cfg['datasets']}")
    if family == "dinov3":
        print(f"  archs         : {cfg['archs']}")
        print(f"  block_types   : {cfg['block_types']}")

    all_jobs = build_labeled_jobs(cfg)
    api = wandb.Api(timeout=60)

    if args.save_csv:
        output_dir.mkdir(parents=True, exist_ok=True)

    for group in groups:
        tag = group if group else family  # arch name, or family for spectre/ctfm
        print(f"\n{'─'*60}\nFetching: {tag}\n{'─'*60}")
        df = fetch_runs(group, cfg, api)
        if df.empty:
            print("  No data.")
            continue

        # --all-methods: rows from every method actually in the shared project;
        # otherwise just this config's model jobs.
        if args.all_methods:
            group_jobs = jobs_from_runs(df, family)
        else:
            group_jobs = [j for j in all_jobs if j["arch"] == group]
        title_base = f"{tag}  |  {mode}"

        for seed in cfg["seeds"]:
            seed_df = df[df["seed"] == seed]
            if seed_df.empty:
                continue
            table = build_table(seed_df, group_jobs, cfg["datasets"])
            print_table(table, f"{title_base}  |  seed={seed}", cfg["datasets"])
            if args.save_csv:
                path = output_dir / f"{tag}_{mode}_seed{seed}.csv"
                table.to_csv(path)
                print(f"  Saved: {path}")

        mean_table = build_table(df, group_jobs, cfg["datasets"])
        print_table(mean_table, f"{title_base}  |  mean over {cfg['seeds']}", cfg["datasets"])
        if args.save_csv:
            path = output_dir / f"{tag}_{mode}_mean.csv"
            mean_table.to_csv(path)
            print(f"  Saved: {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
