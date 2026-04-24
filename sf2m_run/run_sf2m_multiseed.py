#!/usr/bin/env python
"""
Run SF2M across multiple seeds and summarize held-out W1.

For each seed:
1) Runs sf2m_run/run_sf2m.py
2) Reads {save_dir}/w1_scores.tsv
3) Computes mean W1 across held-out timepoints for that run

Then writes a CSV with per-run means and a final summary row containing
mean/std across runs.
"""

import argparse
import csv
import os
import subprocess
import sys

import numpy as np


def parse_seeds(seed_str):
    seeds = []
    for s in seed_str.split(","):
        s = s.strip()
        if not s:
            continue
        seeds.append(int(s))
    if not seeds:
        raise ValueError("No seeds provided.")
    return seeds


def read_w1_values(tsv_path):
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"Missing W1 file: {tsv_path}")
    vals = []
    with open(tsv_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            vals.append(float(row["w1"]))
    if not vals:
        raise ValueError(f"No W1 values found in {tsv_path}")
    return vals


def build_run_cmd(args, seed, save_dir):
    cmd = [
        sys.executable,
        args.run_script,
        "--adata_path",
        args.adata_path,
        "--pc_dim",
        str(args.pc_dim),
        "--t0_index",
        str(args.t0_index),
        "--t1_index",
        str(args.t1_index),
        "--sigma",
        str(args.sigma),
        "--n_iters",
        str(args.n_iters),
        "--batch_size",
        str(args.batch_size),
        "--w",
        str(args.w),
        "--eval_times",
        args.eval_times,
        "--num_traj",
        str(args.num_traj),
        "--ode_steps",
        str(args.ode_steps),
        "--mouse_subset",
        args.mouse_subset,
        "--seed",
        str(seed),
        "--save_dir",
        save_dir,
    ]
    if args.tissue is not None:
        cmd.extend(["--tissue", args.tissue])
    return cmd


def main():
    p = argparse.ArgumentParser(description="Run SF2M over multiple seeds and summarize W1.")
    p.add_argument("--run_script", default="sf2m_run/run_sf2m.py")
    p.add_argument("--adata_path", default="atlas/mouse_preprocessed.h5ad")
    p.add_argument(
        "--tissue",
        default=None,
        help="Optional tissue subset passed to run_sf2m.py (e.g. 'Central Nervous System').",
    )
    p.add_argument("--pc_dim", type=int, default=100)
    p.add_argument("--t0_index", type=int, default=1)
    p.add_argument("--t1_index", type=int, default=4)
    p.add_argument("--sigma", type=float, default=0.25)
    p.add_argument("--n_iters", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--w", type=int, default=64)
    p.add_argument("--eval_times", default="")
    p.add_argument("--num_traj", type=int, default=6000)
    p.add_argument("--ode_steps", type=int, default=400)
    p.add_argument(
        "--mouse_subset",
        default="none",
        choices=["none", "blood", "brain"],
        help="Optional mouse subset to pass through to run_sf2m.py.",
    )
    p.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10")
    p.add_argument("--base_save_dir", default="sf2m_run/outputs/multiseed")
    p.add_argument("--summary_csv", default="sf2m_run/outputs/multiseed_w1_summary.csv")
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()

    seeds = parse_seeds(args.seeds)
    os.makedirs(args.base_save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.summary_csv) or ".", exist_ok=True)

    run_rows = []
    for seed in seeds:
        run_dir = os.path.join(args.base_save_dir, f"seed_{seed}")
        w1_tsv = os.path.join(run_dir, "w1_scores.tsv")

        if args.skip_existing and os.path.exists(w1_tsv):
            print(f"[seed={seed}] Using existing {w1_tsv}")
        else:
            os.makedirs(run_dir, exist_ok=True)
            cmd = build_run_cmd(args, seed, run_dir)
            print(f"[seed={seed}] Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

        w1_vals = read_w1_values(w1_tsv)
        run_mean_w1 = float(np.mean(w1_vals))
        run_rows.append(
            {
                "row_type": "run",
                "seed": seed,
                "run_dir": run_dir,
                "num_heldout_timepoints": len(w1_vals),
                "mean_w1_across_heldout": run_mean_w1,
                "mean_w1_across_runs": "",
                "std_w1_across_runs": "",
                "n_runs": "",
            }
        )
        print(
            f"[seed={seed}] mean_w1_across_heldout={run_mean_w1:.6f} "
            f"(n_timepoints={len(w1_vals)})"
        )

    run_means = np.array([r["mean_w1_across_heldout"] for r in run_rows], dtype=float)
    mean_across_runs = float(np.mean(run_means))
    std_across_runs = float(np.std(run_means, ddof=1 if len(run_means) > 1 else 0))

    summary_row = {
        "row_type": "summary",
        "seed": "",
        "run_dir": "",
        "num_heldout_timepoints": "",
        "mean_w1_across_heldout": "",
        "mean_w1_across_runs": mean_across_runs,
        "std_w1_across_runs": std_across_runs,
        "n_runs": len(run_rows),
    }

    fieldnames = [
        "row_type",
        "seed",
        "run_dir",
        "num_heldout_timepoints",
        "mean_w1_across_heldout",
        "mean_w1_across_runs",
        "std_w1_across_runs",
        "n_runs",
    ]
    with open(args.summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(run_rows)
        writer.writerow(summary_row)

    print(f"Wrote summary CSV: {args.summary_csv}")
    print(f"mean_w1_across_runs={mean_across_runs:.6f}")
    print(f"std_w1_across_runs={std_across_runs:.6f}")


if __name__ == "__main__":
    main()
