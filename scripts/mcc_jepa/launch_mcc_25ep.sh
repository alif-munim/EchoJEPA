#!/bin/bash
# Launch helper for MCC-JEPA experiments.
#
# Usage:
#   scripts/mcc_jepa/launch_mcc_25ep.sh --smoke-only         # just the target-anchored smoke
#   scripts/mcc_jepa/launch_mcc_25ep.sh --smoke-and-pure     # anchored smoke + pure diagnostic
#   scripts/mcc_jepa/launch_mcc_25ep.sh --yes-25ep           # vanilla +25 control + anchored +25
#
# By default prints the sbatch commands without submitting; --yes-25ep is
# required to actually submit the 25-epoch runs.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# 1. Checkpoint sanity.
CKPT="$ROOT/checkpoints/jepa_in21k_vitl_e100.pt"
if [[ ! -f "$CKPT" ]]; then
  echo "[FATAL] missing checkpoint: $CKPT"
  exit 1
fi
echo "[OK] checkpoint present: $CKPT ($(stat -c '%s' "$CKPT") bytes)"

# 2. Config sanity: parse all 4 YAMLs.
CFGS=(
  "configs/train/vitl16/pretrain-vjepa-in21k-e100-plus25-control.yaml"
  "configs/train/vitl16/pretrain-mcc-jepa-pure-smoke.yaml"
  "configs/train/vitl16/pretrain-mcc-jepa-target-anchored-smoke.yaml"
  "configs/train/vitl16/pretrain-mcc-jepa-target-anchored-25of100.yaml"
)
python - <<PY
import sys, yaml
for p in """${CFGS[*]}""".split():
    with open(p) as f:
        cfg = yaml.safe_load(f)
    assert cfg["optimization"]["anneal_ckpt"].endswith("jepa_in21k_vitl_e100.pt"), p
    assert cfg["optimization"]["force_load_pretrain"] is True, p
    print(f"[OK] {p}  app={cfg['app']}  stop_after_epochs={cfg['optimization']['stop_after_epochs']}")
PY

# 3. Tests.
echo "[INFO] pytest tests/mcc_jepa/"
python -m pytest tests/mcc_jepa/ -q

# 4. Sbatch syntax.
for f in scripts/mcc_jepa/*.sbatch; do bash -n "$f"; done
echo "[OK] all sbatches parse"

# 5. Pair-sampler dry-run (tiny synthetic manifest).
python - <<'PY'
import pandas as pd
from src.datasets.mcc_pair_dataset import build_pair_manifest, sampler_diagnostics
rows = []
views = ["A4C","A2C","A3C","A5C","PLAX","PSAX-MV","PSAX-AV","A4C"]
mods = ["bmode"]*7 + ["color"]
for s in range(10):
    for i in range(8):
        rows.append(dict(study_id=f"s{s}", path=f"s{s}_c{i}.mp4", view=views[i], modality=mods[i]))
df = pd.DataFrame(rows)
pair_df = build_pair_manifest(df, seed=7)
diag = sampler_diagnostics(pair_df)
print(f"[OK] pair-sampler dry-run: n={diag['n_pairs']} "
      f"same_study={diag['pair_same_study_rate']:.2f} "
      f"distinct={diag['pair_distinct_clip_rate']:.2f} "
      f"fallback={diag['fallback_fraction']:.2f}")
PY

# 6. Print launch plan.
cat <<EOF

=== MCC-JEPA launch plan ===
Smoke (target-anchored, ~90 min):
  sbatch scripts/mcc_jepa/pretrain_mcc_target_anchored_smoke.sbatch

Smoke (pure, diagnostic only, ~90 min):
  sbatch scripts/mcc_jepa/pretrain_mcc_pure_smoke.sbatch

25-epoch vanilla control:
  sbatch scripts/mcc_jepa/pretrain_vjepa_plus25_control.sbatch

25-epoch target-anchored MCC:
  sbatch scripts/mcc_jepa/pretrain_mcc_target_anchored_25ep.sbatch

Gate the 25-epoch runs on the target-anchored smoke passing all gates
listed in reports/mcc_jepa/launch_readiness.md.
EOF

if [[ "${1-}" == "--smoke-only" ]]; then
  echo "--smoke-only: run the target-anchored smoke sbatch by hand when ready."
  exit 0
fi

if [[ "${1-}" == "--yes-25ep" ]]; then
  echo "--yes-25ep flag set: submit both 25-epoch jobs now."
  echo "[SUBMIT] vanilla +25 control"
  sbatch scripts/mcc_jepa/pretrain_vjepa_plus25_control.sbatch
  echo "[SUBMIT] target-anchored +25"
  sbatch scripts/mcc_jepa/pretrain_mcc_target_anchored_25ep.sbatch
  exit 0
fi

echo ""
echo "Re-run with --smoke-only, --smoke-and-pure, or --yes-25ep to submit jobs."
