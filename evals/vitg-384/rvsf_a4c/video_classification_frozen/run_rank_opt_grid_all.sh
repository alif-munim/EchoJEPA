#!/usr/bin/env bash
set -euo pipefail

# Run from the directory that contains rank_opt_grid.py (or adjust path below)
SCRIPT="./rank_opt_grid.py"

# List your checkpoints (relative paths are fine)
CKPTS=(
"mrvsf-vitg16-336-16f-pt-e279-an-e78-fs1-ns3-nvs2/epoch_001.pt"
"mrvsf-vitg16-336-16f-pt-e279-an-e78-fs1-ns3-nvs2/epoch_002.pt"
"rvsf-a4c-full-vitg16-336-16f-pt-e279-an-e78-fs1-ns3-nvs2/epoch_001.pt"
"rvsf-a4c-full-vitg16-336-16f-pt-e279-an-e78-fs1-ns3-nvs2/latest.pt"
"rvsf-a4c-full-vitg16-336-16f-pt280-an80-fs1-ns2-nvs2/epoch_001.pt"
"rvsf-a4c-full-vitg16-336-16f-pretrain-e279-fs1-ns3-nvs2/epoch_001.pt"
"rvsf-a4c-full-vitg16-336-16f-pretrain-e279-fs1-ns3-nvs2/epoch_002.pt"
"rvsf-a4c-full-vitg16-336-16f-pretrain-e279-fs1-ns3-nvs2/epoch_003.pt"
"rvsf-a4c-full-vitg16-336-16f-pretrain-e279-fs1-ns3-nvs2/latest.pt"
)

METRIC="best_val_acc_per_head"
TOPK=10

for ckpt in "${CKPTS[@]}"; do
  echo "================================================================================"
  echo "CKPT: $ckpt"
  python3 "$SCRIPT" \
    --ckpt "$ckpt" \
    --metric "$METRIC" \
    --topk "$TOPK" \
    --print-grid-dicts \
    --device meta || true
done
