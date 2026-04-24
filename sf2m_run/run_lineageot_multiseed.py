#!/usr/bin/env python
"""
Run LineageOT across multiple seeds and summarize held-out W1.

LineageOT is deterministic given the data, so results are identical across
seeds.  This wrapper exists for consistency with the SF2M multiseed interface
and to produce a summary CSV in the same format as
run_sf2m_multiseed.py.

Usage (from finfm project root):
    python sf2m_run/run_lineageot_multiseed.py --dataset mouse_blood
    python sf2m_run/run_lineageot_multiseed.py --dataset zebrafish_cns \\
        --seeds 1,2,3 --summary_csv sf2m_run/outputs/lineageot_zebrafish_summary.csv
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
        if s:
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
        "--dataset", args.dataset,
        "--adata_path", args.adata_path,
        "--pc_dim", str(args.pc_dim),
        "--t0_index", str(args.t0_index),
        "--t1_index", str(args.t1_index),
        "--epsilon", str(args.epsilon),
        "--root_time_factor", str(args.root_time_factor),
        "--paga_threshold", str(args.paga_threshold),
        "--seed", str(seed),
        "--save_dir", save_dir,
    ]
    return cmd


def main():
    p = argparse.ArgumentParser(
        description="Run LineageOT over multiple seeds and summarize W1."
    )
    p.add_argument("--run_script", default="sf2m_run/run_lineageot.py")
    p.add_argument(
        "--dataset", required=True, choices=["mouse_blood", "zebrafish_cns", "zebrafish_pa"],
        help="Dataset to benchmark",
    )
    p.add_argument("--adata_path", default=None)
    p.add_argument("--pc_dim", type=int, default=100)
    p.add_argument("--t0_index", type=int, default=None)
    p.add_argument("--t1_index", type=int, default=None)
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--root_time_factor", type=float, default=1000)
    p.add_argument("--paga_threshold", type=float, default=0.1)
    p.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10")
    p.add_argument("--base_save_dir", default="sf2m_run/outputs/lineageot_multiseed")
    p.add_argument(
        "--summary_csv",
        default=None,
        help="Output summary CSV path.  Default: sf2m_run/outputs/lineageot_{dataset}_summary.csv",
    )
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()

    # Fill dataset-specific defaults
    if args.adata_path is None:
        args.adata_path = (
            "atlas/mouse_preprocessed.h5ad"
            if args.dataset == "mouse_blood"
            else "zebrafish_neural.h5ad"
        )
    if args.t0_index is None:
        args.t0_index = 1
    if args.t1_index is None:
        args.t1_index = 7 if args.dataset == "mouse_blood" else 4
    if args.summary_csv is None:
        args.summary_csv = (
            f"sf2m_run/outputs/lineageot_{args.dataset}_summary.csv"
        )

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
        run_mean = float(np.mean(w1_vals))
        run_rows.append(
            {
                "row_type": "run",
                "seed": seed,
                "run_dir": run_dir,
                "num_heldout_timepoints": len(w1_vals),
                "mean_w1_across_heldout": run_mean,
                "mean_w1_across_runs": "",
                "std_w1_across_runs": "",
                "n_runs": "",
            }
        )
        print(
            f"[seed={seed}] mean_w1_across_heldout={run_mean:.6f} "
            f"(n_timepoints={len(w1_vals)})"
        )

    run_means = np.array([r["mean_w1_across_heldout"] for r in run_rows], dtype=float)
    mean_across = float(np.mean(run_means))
    std_across = float(np.std(run_means, ddof=1 if len(run_means) > 1 else 0))

    summary_row = {
        "row_type": "summary",
        "seed": "",
        "run_dir": "",
        "num_heldout_timepoints": "",
        "mean_w1_across_heldout": "",
        "mean_w1_across_runs": mean_across,
        "std_w1_across_runs": std_across,
        "n_runs": len(run_rows),
    }

    fieldnames = [
        "row_type", "seed", "run_dir", "num_heldout_timepoints",
        "mean_w1_across_heldout", "mean_w1_across_runs", "std_w1_across_runs", "n_runs",
    ]
    with open(args.summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(run_rows)
        writer.writerow(summary_row)

    print(f"\nWrote summary CSV: {args.summary_csv}")
    print(f"mean_w1_across_runs={mean_across:.6f}")
    print(f"std_w1_across_runs ={std_across:.6f}")
    print("(Note: LineageOT is deterministic — std across seeds should be ~0.)")


if __name__ == "__main__":
    main()
