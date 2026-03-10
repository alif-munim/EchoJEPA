#!/bin/bash
# Verification experiment — Instance 1
# EchoJEPA-G + EchoMAE-L (ViT-based, 1568 tokens) at depth=1 and depth=4
# Plus EchoPrime + PanEcho at depth=4
set -euo pipefail

cd /mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2
CONFIG=configs/eval/vitg-384/view/verification
DEVICES="cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7"

echo "=== Instance 1: 6 runs ==="
echo ""

echo ">>> [1/6] EchoJEPA-G depth=1..."
PYTHONUNBUFFERED=1 python -m evals.main --fname $CONFIG/echojepa_g_d1.yaml --devices $DEVICES
echo ">>> DONE"
echo ""

echo ">>> [2/6] EchoJEPA-G depth=4..."
PYTHONUNBUFFERED=1 python -m evals.main --fname $CONFIG/echojepa_g_d4.yaml --devices $DEVICES
echo ">>> DONE"
echo ""

echo ">>> [3/6] EchoMAE-L depth=1..."
PYTHONUNBUFFERED=1 python -m evals.main --fname $CONFIG/echomae_d1.yaml --devices $DEVICES
echo ">>> DONE"
echo ""

echo ">>> [4/6] EchoMAE-L depth=4..."
PYTHONUNBUFFERED=1 python -m evals.main --fname $CONFIG/echomae_d4.yaml --devices $DEVICES
echo ">>> DONE"
echo ""

echo ">>> [5/6] EchoPrime depth=4..."
PYTHONUNBUFFERED=1 python -m evals.main --fname $CONFIG/echoprime_d4.yaml --devices $DEVICES
echo ">>> DONE"
echo ""

echo ">>> [6/6] PanEcho depth=4..."
PYTHONUNBUFFERED=1 python -m evals.main --fname $CONFIG/panecho_d4.yaml --devices $DEVICES
echo ">>> DONE"
echo ""

echo "=== Instance 1 complete (6/6) ==="
