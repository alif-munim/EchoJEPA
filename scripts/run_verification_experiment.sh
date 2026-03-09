#!/bin/bash
# Verification experiment: Confirm depth=1 attentive probes >= linear for all architectures
#
# Runs 4 models x 2 depths = 8 configs on UHN 13-class view classification.
# Each config trains 20 probes in parallel (5 LR x 4 WD via multihead_kwargs).
# See context_files/probe_implementation_analysis.md for the full experiment spec.
#
# Usage:
#   bash scripts/run_verification_experiment.sh          # depth=1 only (shortcut, ~32 GPU-hours)
#   bash scripts/run_verification_experiment.sh --full    # depth=1 + depth=4 (~64 GPU-hours)
#
# Prerequisites:
#   - 8x A100 GPUs
#   - Fixed normalization code (post-commit 4803640)
#   - EchoPrime model at modelcustom/EchoPrime/
#   - PanEcho model (downloads from hub)
#   - EchoJEPA-G checkpoint at checkpoints/anneal/keep/pt-280-an81.pt
#   - EchoMAE-L checkpoint at checkpoints/videomae-ep163.pth

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$REPO_ROOT/configs/eval/vitg-384/view/verification"

# Parse args
FULL_RUN=false
if [[ "${1:-}" == "--full" ]]; then
    FULL_RUN=true
fi

MODELS=(echojepa_g echomae echoprime panecho)
LINEAR_MODELS=(echoprime panecho)  # Need re-run with fixed normalization

echo "============================================"
echo "Attentive Probe Verification Experiment"
echo "============================================"
echo "Models: ${MODELS[*]}"
if $FULL_RUN; then
    echo "Depths: 1, 4 (full run)"
else
    echo "Depths: 1 only (shortcut)"
fi
echo "Linear baselines (fixed norm): ${LINEAR_MODELS[*]}"
echo "Config dir: $CONFIG_DIR"
echo "============================================"
echo ""

# Step 1: Run linear baselines for EchoPrime/PanEcho (need correct normalization baseline)
echo "=== Step 1: Linear baselines (fixed normalization) ==="
for model in "${LINEAR_MODELS[@]}"; do
    config="$CONFIG_DIR/${model}_linear.yaml"
    if [[ ! -f "$config" ]]; then
        echo "SKIP: $config not found"
        continue
    fi

    echo ">>> Running $model linear baseline..."
    PYTHONUNBUFFERED=1 python -m evals.main \
        --fname "$config" \
        --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
    echo ">>> $model linear DONE"
    echo ""
done

# Step 2: Run depth=1 attentive (primary verification)
echo "=== Step 2: Depth=1 attentive probes ==="
for model in "${MODELS[@]}"; do
    config="$CONFIG_DIR/${model}_d1.yaml"
    if [[ ! -f "$config" ]]; then
        echo "ERROR: Config not found: $config"
        continue
    fi

    echo ">>> Running $model depth=1..."
    PYTHONUNBUFFERED=1 python -m evals.main \
        --fname "$config" \
        --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
    echo ">>> $model depth=1 DONE"
    echo ""
done

# Step 3: Run depth=4 (comparison, only if --full)
if $FULL_RUN; then
    echo "=== Step 3: Depth=4 attentive probes ==="
    for model in "${MODELS[@]}"; do
        config="$CONFIG_DIR/${model}_d4.yaml"
        if [[ ! -f "$config" ]]; then
            echo "ERROR: Config not found: $config"
            continue
        fi

        echo ">>> Running $model depth=4..."
        PYTHONUNBUFFERED=1 python -m evals.main \
            --fname "$config" \
            --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
        echo ">>> $model depth=4 DONE"
        echo ""
    done
fi

echo "============================================"
echo "Verification experiment complete."
echo "Results at: $REPO_ROOT/evals/vitg-384/classifier/"
echo ""
echo "Decision criteria:"
echo "  EchoPrime d=1 attentive >= 55% (within 3pp of linear ~58%)"
echo "    => Strategy E: uniform depth=1 attentive for all models"
echo "  EchoPrime d=1 attentive < 50% (>5pp below linear)"
echo "    => Strategy B: linear primary + attentive JEPA ceiling"
echo "============================================"
