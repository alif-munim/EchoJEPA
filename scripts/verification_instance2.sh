#!/bin/bash
# Verification experiment — Instance 2
# EchoPrime + PanEcho linear baselines (fixed normalization)
# EchoPrime + PanEcho depth=1 attentive (THE KEY TEST)
# EchoJEPA-G + EchoMAE-L linear baselines for completeness
set -euo pipefail

cd /mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2
CONFIG=configs/eval/vitg-384/view/verification
DEVICES="cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7"

echo "=== Instance 2: 6 runs ==="
echo ""

echo ">>> [1/6] EchoPrime LINEAR (fixed normalization baseline)..."
PYTHONUNBUFFERED=1 python -m evals.main --fname $CONFIG/echoprime_linear.yaml --devices $DEVICES
echo ">>> DONE"
echo ""

echo ">>> [2/6] PanEcho LINEAR (fixed normalization baseline)..."
PYTHONUNBUFFERED=1 python -m evals.main --fname $CONFIG/panecho_linear.yaml --devices $DEVICES
echo ">>> DONE"
echo ""

echo ">>> [3/6] EchoPrime ATTENTIVE depth=1 (DECISION-CRITICAL)..."
PYTHONUNBUFFERED=1 python -m evals.main --fname $CONFIG/echoprime_d1.yaml --devices $DEVICES
echo ">>> DONE"
echo ""

echo ">>> [4/6] PanEcho ATTENTIVE depth=1..."
PYTHONUNBUFFERED=1 python -m evals.main --fname $CONFIG/panecho_d1.yaml --devices $DEVICES
echo ">>> DONE"
echo ""

echo ">>> [5/6] EchoJEPA-G LINEAR (reference)..."
PYTHONUNBUFFERED=1 python -m evals.main --fname $CONFIG/../echojepa_view_linear.yaml --devices $DEVICES
echo ">>> DONE"
echo ""

echo ">>> [6/6] EchoMAE-L LINEAR (reference — uses VideoMAE encoder)..."
# Note: no dedicated EchoMAE linear config exists yet.
# If you have one, replace the path below. Otherwise skip this run.
echo ">>> SKIPPED — need to create echomae_linear.yaml config"
echo ""

echo "=== Instance 2 complete ==="
echo ""
echo "DECISION: Compare these pairs:"
echo "  EchoPrime:  verify-echoprime-d1 vs verify-echoprime-linear"
echo "  PanEcho:    verify-panecho-d1 vs verify-panecho-linear"
echo "  EchoJEPA-G: verify-echojepa-g-d1 (instance 1) vs echojepa-view-linear (this instance)"
echo ""
echo "  All attentive >= linear => Strategy E (uniform depth=1 attentive)"
