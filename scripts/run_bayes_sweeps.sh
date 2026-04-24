#!/usr/bin/env bash
# For each Bayesian sweep config: run the Bayes phase, then automatically
# run a multi-seed grid over the best parameters found.
#
# Usage:
#   bash scripts/run_bayes_sweeps.sh
#   bash scripts/run_bayes_sweeps.sh --dataset mouse_blood
#   bash scripts/run_bayes_sweeps.sh --dataset cite --count 30 --seeds 5
#   bash scripts/run_bayes_sweeps.sh mfm_bayes mfm_finsler_bayes   # specific configs
#
# Positional args (optional): config names without .yaml extension.
# If none given, all configs in ALL_SWEEPS are run.

set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
DATASET="zebrafish_cns"
COUNT_ARG=""
SEEDS=10
SWEEP_DIR="configs/sweeps"

ALL_SWEEPS=(mfm_bayes mfm_finsler_bayes)

# ── parse flags ───────────────────────────────────────────────────────────────
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --count)   COUNT_ARG="--count $2"; shift 2 ;;
        --seeds)   SEEDS="$2"; shift 2 ;;
        --*)       echo "Unknown flag: $1" >&2; exit 1 ;;
        *)         POSITIONAL+=("$1"); shift ;;
    esac
done

if [[ ${#POSITIONAL[@]} -gt 0 ]]; then
    SWEEPS=("${POSITIONAL[@]}")
else
    SWEEPS=("${ALL_SWEEPS[@]}")
fi

# ── run ───────────────────────────────────────────────────────────────────────
echo "Dataset  : $DATASET"
echo "Configs  : ${SWEEPS[*]}"
echo "Seeds    : $SEEDS"
echo "────────────────────────────────────────────────────────────────────────"

for name in "${SWEEPS[@]}"; do
    cfg="$SWEEP_DIR/$name.yaml"
    echo ""
    echo "=== BAYES PHASE: $name ($cfg) ==="

    # Run the Bayes sweep; capture stdout to a temp file while still showing it
    tmpfile=$(mktemp)
    python -m scripts.sweep \
        --sweep_config "$cfg" \
        --dataset "$DATASET" \
        $COUNT_ARG 2>&1 | tee "$tmpfile"

    # Extract the sweep ID printed by sweep.py
    sweep_id=$(grep '^SWEEP_ID=' "$tmpfile" | tail -1 | cut -d= -f2)
    rm "$tmpfile"

    if [[ -z "$sweep_id" ]]; then
        echo "ERROR: Could not capture sweep ID from $name. Skipping multi-seed phase." >&2
        continue
    fi

    echo ""
    echo "=== SEED PHASE: $name → sweep $sweep_id ($SEEDS seeds) ==="
    python -m scripts.sweep \
        --from_sweep "$sweep_id" \
        --dataset "$DATASET" \
        --seeds "$SEEDS"
done

echo ""
echo "All Bayes sweeps complete."
