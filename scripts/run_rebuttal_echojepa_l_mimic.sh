#!/bin/bash
# Sequential chain: EchoJEPA-L (MIMIC) rebuttal probes — LVEF, view, RVSP
# Rebuttal subsets (10K/5K/5K train, 1K val), d=4, 20 epochs
# Expected: ~1.7h LVEF + ~50min view + ~50min RVSP = ~3.3h total

set -e
export TMPDIR=/tmp
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH

CD=/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2
DEVICES="cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7"
LOGDIR=/home/sagemaker-user/user-default-efs/vjepa2/logs

cd $CD

echo "=== [$(date)] Starting LVEF ===" | tee -a $LOGDIR/rebuttal_chain.log
python -m evals.main --fname configs/eval/vitl/icml/echojepa_l_mimic_lvef_d4.yaml --devices $DEVICES \
  2>&1 | tee $LOGDIR/icml_l_mimic_lvef_d4.log
echo "=== [$(date)] LVEF done ===" | tee -a $LOGDIR/rebuttal_chain.log

echo "=== [$(date)] Starting View ===" | tee -a $LOGDIR/rebuttal_chain.log
python -m evals.main --fname configs/eval/vitl/icml/echojepa_l_mimic_view_d4.yaml --devices $DEVICES \
  2>&1 | tee $LOGDIR/icml_l_mimic_view_d4.log
echo "=== [$(date)] View done ===" | tee -a $LOGDIR/rebuttal_chain.log

echo "=== [$(date)] Starting RVSP ===" | tee -a $LOGDIR/rebuttal_chain.log
python -m evals.main --fname configs/eval/vitl/icml/echojepa_l_mimic_rvsp_d4.yaml --devices $DEVICES \
  2>&1 | tee $LOGDIR/icml_l_mimic_rvsp_d4.log
echo "=== [$(date)] RVSP done ===" | tee -a $LOGDIR/rebuttal_chain.log

echo "=== [$(date)] ALL DONE ===" | tee -a $LOGDIR/rebuttal_chain.log
