#!/bin/bash
# Overnight Node 1 — Verification experiment (8×A100)
# 11 runs on UHN 22K view classification (S3)
# Estimated: ~3-4 hours
set -euo pipefail

cd /mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2
DEVICES="cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7"
V=configs/eval/vitg-384/view/verification

START=$(date +%s)
log() { echo "[$(date '+%H:%M:%S')] $1"; }
run() { log ">>> $1..."; PYTHONUNBUFFERED=1 python -m evals.main --fname "$2" --devices $DEVICES; log ">>> DONE"; echo ""; }

log "=== NODE 1: VERIFICATION (11 runs) ==="

# Linear baselines (fixed normalization)
run "[1/11] EchoPrime LINEAR"            $V/echoprime_linear.yaml
run "[2/11] PanEcho LINEAR"              $V/panecho_linear.yaml

# Create a fresh EchoJEPA-G linear config with resume_checkpoint: false
cp configs/eval/vitg-384/view/echojepa_view_linear.yaml /tmp/echojepa_g_linear_fresh.yaml
sed -i 's/resume_checkpoint: true/resume_checkpoint: false/' /tmp/echojepa_g_linear_fresh.yaml
sed -i 's/tag:.*/tag: verify-echojepa-g-linear/' /tmp/echojepa_g_linear_fresh.yaml
run "[3/11] EchoJEPA-G LINEAR"           /tmp/echojepa_g_linear_fresh.yaml

# Depth=1 attentive
run "[4/11] EchoPrime depth=1 (KEY)"     $V/echoprime_d1.yaml
run "[5/11] PanEcho depth=1"             $V/panecho_d1.yaml
run "[6/11] EchoJEPA-G depth=1"          $V/echojepa_g_d1.yaml
run "[7/11] EchoMAE-L depth=1"           $V/echomae_d1.yaml

# Depth=4 attentive
run "[8/11] EchoPrime depth=4"           $V/echoprime_d4.yaml
run "[9/11] PanEcho depth=4"             $V/panecho_d4.yaml
run "[10/11] EchoJEPA-G depth=4"         $V/echojepa_g_d4.yaml
run "[11/11] EchoMAE-L depth=4"          $V/echomae_d4.yaml

END=$(date +%s)
echo ""
log "=== NODE 1 DONE: $(( (END-START)/60 )) min ==="
echo ""
echo "DECISION — compare on view classification accuracy:"
echo "  EchoPrime: verify-echoprime-d1 vs verify-echoprime-linear"
echo "  PanEcho:   verify-panecho-d1 vs verify-panecho-linear"
echo "  EchoJEPA:  verify-echojepa-g-d1 vs verify-echojepa-g-linear"
echo ""
echo "  attentive >= linear for all => Strategy E"
echo ""
echo "DEPTH — compare d=1 vs d=4:"
echo "  Expected: d=4 adds +1-2pp (V-JEPA 2 Table 18)"
