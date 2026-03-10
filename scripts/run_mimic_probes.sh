#!/bin/bash
# Run all MIMIC probes: HP selection on val, evaluation on test.
# 23 tasks × 7 models = 161 runs.

set -e
cd /mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2

MIMIC_DIR="experiments/nature_medicine/mimic"
OUT_DIR="results/probes/nature_medicine/mimic"

MODELS="echojepa_g echojepa_l echojepa_l_kinetics echomae echoprime panecho echofm"

CLS_TASKS="discharge_destination disease_afib disease_amyloidosis disease_dcm disease_hcm disease_hf disease_stemi disease_takotsubo disease_tamponade drg_severity icu_transfer in_hospital_mortality mortality_1yr mortality_30d mortality_90d readmission_30d triage_acuity"

REG_TASKS="creatinine ef_note_extracted lactate los_remaining nt_probnp troponin_t"

TOTAL=0
DONE=0
FAILED=0

for MODEL in $MODELS; do
    SPLIT_DIR="${MIMIC_DIR}/${MODEL}_splits"
    if [ ! -d "$SPLIT_DIR" ]; then
        echo "SKIP $MODEL: no splits directory"
        continue
    fi

    # Classification tasks
    for TASK in $CLS_TASKS; do
        TRAIN="${SPLIT_DIR}/${TASK}/train.npz"
        VAL="${SPLIT_DIR}/${TASK}/val.npz"
        TEST="${SPLIT_DIR}/${TASK}/test.npz"
        OUTD="${OUT_DIR}/${MODEL}/${TASK}"

        if [ ! -f "$TRAIN" ]; then
            echo "SKIP $MODEL/$TASK: no train.npz"
            continue
        fi

        TOTAL=$((TOTAL + 1))

        # Skip if already done
        if [ -f "${OUTD}/test_metrics.json" ]; then
            echo "SKIP $MODEL/$TASK: already done"
            DONE=$((DONE + 1))
            continue
        fi

        echo "=== $MODEL / $TASK (classification) ==="
        mkdir -p "$OUTD"

        # HP selection on val
        python -m evals.train_probe \
            --train "$TRAIN" --val "$VAL" \
            --task classification \
            --output_dir "${OUTD}" 2>&1 | tail -3

        # Evaluate best model on test
        python -c "
import numpy as np, json, joblib, os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score, f1_score

outd = '${OUTD}'
model = joblib.load(os.path.join(outd, 'train', 'model.joblib'))
scaler = joblib.load(os.path.join(outd, 'train', 'scaler.joblib'))

test = np.load('${TEST}', allow_pickle=True)
X_test = scaler.transform(test['embeddings'])
y_test = test['labels']

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

n_classes = len(np.unique(y_test))
if n_classes == 2:
    auc = roc_auc_score(y_test, y_proba[:, 1])
else:
    auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')

metrics = {
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'balanced_accuracy': float(balanced_accuracy_score(y_test, y_pred)),
    'auc': float(auc),
    'f1_macro': float(f1_score(y_test, y_pred, average='macro')),
    'n_test': int(len(y_test)),
    'n_classes': int(n_classes),
}
with open(os.path.join(outd, 'test_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)
print(f'  TEST: AUC={auc:.3f}, BalAcc={metrics[\"balanced_accuracy\"]:.3f}, n={len(y_test)}')
"
        DONE=$((DONE + 1))
    done

    # Regression tasks
    for TASK in $REG_TASKS; do
        TRAIN="${SPLIT_DIR}/${TASK}/train.npz"
        VAL="${SPLIT_DIR}/${TASK}/val.npz"
        TEST="${SPLIT_DIR}/${TASK}/test.npz"
        OUTD="${OUT_DIR}/${MODEL}/${TASK}"

        if [ ! -f "$TRAIN" ]; then
            echo "SKIP $MODEL/$TASK: no train.npz"
            continue
        fi

        TOTAL=$((TOTAL + 1))

        if [ -f "${OUTD}/test_metrics.json" ]; then
            echo "SKIP $MODEL/$TASK: already done"
            DONE=$((DONE + 1))
            continue
        fi

        echo "=== $MODEL / $TASK (regression) ==="
        mkdir -p "$OUTD"

        # HP selection on val
        python -m evals.train_probe \
            --train "$TRAIN" --val "$VAL" \
            --task regression \
            --output_dir "${OUTD}" 2>&1 | tail -3

        # Evaluate best model on test
        python -c "
import numpy as np, json, joblib, os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr

outd = '${OUTD}'
model = joblib.load(os.path.join(outd, 'train', 'model.joblib'))
scaler = joblib.load(os.path.join(outd, 'train', 'scaler.joblib'))

test = np.load('${TEST}', allow_pickle=True)
X_test = scaler.transform(test['embeddings'])
y_test = test['labels']

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r, p = pearsonr(y_test, y_pred)

metrics = {
    'r2': float(r2),
    'mae': float(mae),
    'pearson_r': float(r),
    'pearson_p': float(p),
    'n_test': int(len(y_test)),
}
with open(os.path.join(outd, 'test_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)
print(f'  TEST: R2={r2:.3f}, MAE={mae:.3f}, r={r:.3f}, n={len(y_test)}')
"
        DONE=$((DONE + 1))
    done
done

echo ""
echo "========================================="
echo "COMPLETE: $DONE/$TOTAL runs finished, $FAILED failed"
echo "========================================="
