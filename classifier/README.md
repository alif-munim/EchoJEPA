pkill -u sagemaker-user python

# Train ConvNeXT Classifier

First, process your images and generate the training CSV in the `data_prep_v3.ipynb` notebook. It will create the file `labels_masked_inplace.csv`. Then, start your training run.

```
# 0. Activate your environment
conda activate echomamba4

# 1. Define your target directory variable
RESULTS="results/experiment_masked_reg_v2"

# 2. Create it manually first
mkdir -p $RESULTS
```

# First Frame (Default Data)
Train convnext base
```
# 3. Run and pipe to tee (saves to file AND shows on screen)
# 2>&1 redirects errors to stdout so they get saved too
python convnext_training.py \
  --csv_file labels_masked_inplace.csv \
  --weight_decay 0.05 \
  --drop_path 0.2 \
  --patience 8 \
  --epochs 30 \
  --results_dir $RESULTS 2>&1 | tee $RESULTS/output.log
```

Train convnext small
```
python convnext_training.py \
    --results_dir "results/convnext_small_tta_random_gs" \
    --weights "convnext_small_384.bin" \
    --epochs 50 \
    --patience 15
```

Train convnext base
```
python convnext_training.py \
    --results_dir "results/convnext_base_heavy_reg" \
    --weights "convnext_base_384.bin" \
    --model_name "convnext_base.fb_in22k_ft_in1k_384" \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-4 \
    --weight_decay 0.1 \
    --drop_path 0.5 \
    --mixup 0.8 \
    --patience 15
```

Cooldown small
```
python cooldown.py \
    --results_dir "results/cooldown_final" \
    --start_weights "/cluster/home/t115318uhn/echo-reports/echo-view-classifier/cvat_1015_img/results/convnext_small_tta_random_gs/best_model.pth"
```

Cooldown base
```
python cooldown.py \
    --results_dir "results/cooldown_base" \
    --start_weights "results/convnext_base_heavy_reg/best_model.pth" \
    --model_name "convnext_base.fb_in22k_ft_in1k_384" \
    --lr 1e-6 \
    --epochs 10 \
    --batch_size 32
```

Train swin
```
python convnext_training.py \
    --results_dir "results/swin_base_batch96" \
    --weights "swin_base_384.bin" \
    --model_name "swin_base_patch4_window12_384.ms_in22k_ft_in1k" \
    --epochs 50 \
    --batch_size 96 \
    --lr 1.5e-4 \
    --weight_decay 0.05 \
    --drop_path 0.4 \
    --mixup 0.8 \
    --patience 15
```

Didn't bother cooling down Swin, hit a performance ceiling here. Need to randomly select frames rather than just using the first one (which is possibly garbage quality).

| Run   | Model & Architecture | Key Strategy                                                     | Best Validation F1 | Final Test F1 | Status                                                                      |
| ----- | -------------------- | ---------------------------------------------------------------- | ------------------ | ------------- | --------------------------------------------------------------------------- |
| Run 1 | ConvNeXt-Base (Raw)  | Baseline, Low Reg.                                               | 0.839              | 0.801         | Failed. Severe overfitting (Train ≫ Test).                                  |
| Run 2 | ConvNeXt-Small (v2)  | High Reg. (DropPath 0.4), Mixup, Grayscale, Batch 128/LR 2e−4.   | 0.863              | 0.835         | Success. Overfitting resolved; highest initial F1.                          |
| Run 3 | ConvNeXt-Small (v3)  | Cooldown (LR 1e−6, No Mixup, No Grayscale).                      | 0.844              | 0.824         | Polished. Retained robustness, confirmed ceiling.                           |
| Run 4 | ConvNeXt-Base (v3)   | Extreme Reg. (DropPath 0.5), Mixup, Grayscale, Batch 64/LR 1e−4. | 0.864              | 0.819         | Capacity Test. Reached highest Val F1, but failed to generalize it to Test. |
| Run 5 | Swin-Base            | Architecture Switch (from CNN to Transformer), Batch 32/LR 5e−5. | 0.855              | 0.83          | New Ceiling. Highest raw Test F1, but struggled with A5C errors.            |

With EchoJEPA, I was able to bump the test F1 up to 0.877 (224px). Still not perfect, but better than image baselines.

# Multi-Frame (Requires Videos)

Compile a list of studies to download (607)
```
python make_selected_study_dirs.py \
  --labels labels_masked_inplace.csv \
  --es es.txt \
  --es1 es1.txt \
  --es2 es2.txt \
  --out selected_study_dirs.csv
```

Download the studies from S3 (ignoring ones that are already downloaded)
```
python download_selected_studies.py \
  --csv selected_study_dirs_607.csv \
  --dest "/Users/alifmunim/Documents/Classification/uhn_studies_22k_607" \
  --log download.log \
  --only-show-errors \
  --skip-if-exists
```

# Create Patient Splits

The two deidentification files (to map deidentified study IDs to original study IDs, and thereafter the patient IDs to create the splits) are `heartlab-deid.dedup.csv` and `syngo-deid.dedup.csv`. 

Upon checking these, they are actually the same, but the syngo set has more metadata (`DeidentifiedStudyID`,`OriginalStudyID`,`STUDY_REF`,`PATIENT_ID`,`STUDY_DATE`,`STUDY_TIME`,`ACCESSION_NUMBER`) compared to heartlab (`DeidentifiedStudyID`,`OriginalStudyID`). So we will stick to `syngo-deid.dedup.csv` for the splits. 

```
python3 make_patient_split.py \
  --labels labels_masked_inplace.csv \
  --syngo syngo-deid.dedup.csv \
  --out labels_patient_split.csv \
  --seed 42 \
  --train 0.80 --val 0.10 --test 0.10 \
  --min-val 45 --min-test 45 \
  --max-tries 500
```

You will get a `split_summary.txt` file that summarizes the data distribution in each split. This script still maps to the still image extracted from the first frame of each video (e.g. paths like `./unmasked/job_1734/images/h4h_ef_frames/1.2.276.0.7230010.3.1.2.845494328.1.1703598471.21532689/1.2.276.0.7230010.3.1.3.845494328.1.1703598471.21532690/mp4/1.2.276.0.7230010.3.1.4.811753780.1.1703598679.15799699_masked.jpg`), so it needs to be cleaned up. All of the raw 584 patients (585 studies) are stored at `/cluster/projects/bwanggroup/echo_reports/uhn_studies_22k_585/`.

To map to the actual mp4 directories, use this script (additional args include `--check-exists` and `--drop-missing`
```
python3 map_labels_to_mp4.py \
  --in labels_patient_split.csv \
  --root /cluster/projects/bwanggroup/echo_reports/uhn_studies_22k_607/ \
  --out labels_patient_split_mp4.csv
```

Now, we have a dataset file with valid splits ready to train!


Rewrite the old cluster paths to the S3 uris:
```
python3 rewrite_mp4_paths_to_s3.py \
  --in labels_patient_split_mp4.csv \
  --out labels_patient_split_mp4_s3.csv \
  --s3-prefix s3://echodata25/results/uhn_studies_22k_607 \
  --root-marker uhn_studies_22k_607
```

Randomly sample S3 paths, ensure they exist, and verify video shapes:
```
python3 sample_and_check_s3_mp4s.py \
  --csv labels_patient_split_mp4_s3.csv \
  --n 100 \
  --seed 0 \
  --out-prefix check100
```

Create train, test, val splits in JEPA data format:
```
python3 make_view_labels_space_sep.py \
  --in labels_patient_split_mp4_s3.csv \
  --out ../data/csv/uhn_views_22k_train_336px.csv \
  --mapping uhn_views_22k_mapping_train.txt \
  --split train

python3 make_view_labels_space_sep.py \
  --in labels_patient_split_mp4_s3.csv \
  --out ../data/csv/uhn_views_22k_test_336px.csv \
  --mapping uhn_views_22k_mapping_test.txt \
  --split test

python3 make_view_labels_space_sep.py \
  --in labels_patient_split_mp4_s3.csv \
  --out ../data/csv/uhn_views_22k_val_336px.csv \
  --mapping uhn_views_22k_mapping_val.txt \
  --split val
```

### View Classification

Run training
```
python -m evals.main \
    --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/eval/vitg-384/view/echojepa_view_classification_336px.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee view_336px_dec29v3_celoss.log
```

### Data Efficiency

This is the statistically accurate floor that maintains your exact class distribution while guaranteeing that even your rarest view has enough examples for the linear probe to draw a decision boundary.
```
# --- 1% Data Efficiency (Few-Shot) ---
python3 make_stratified_subset.py \
  --input ../data/csv/uhn_views_22k_train.csv \
  --out ../data/csv/uhn_views_22k_train_1percent.csv \
  --percent 1.0 \
  --min 3 \
  --seed 42

# --- 10% Data Efficiency ---
python3 make_stratified_subset.py \
  --input ../data/csv/uhn_views_22k_train.csv \
  --out ../data/csv/uhn_views_22k_train_10percent.csv \
  --percent 10.0 \
  --min 3 \
  --seed 42

# --- 50% Data Efficiency ---
python3 make_stratified_subset.py \
  --input ../data/csv/uhn_views_22k_train.csv \
  --out ../data/csv/uhn_views_22k_train_50percent.csv \
  --percent 50.0 \
  --min 3 \
  --seed 42
```

# Color Classification
~99% accuracy
[   30] train: 97.892%  val(max-head): 98.769%

From the notebook, we have `labels_color.csv`, `labels_quality.csv`, and `labels_zoom.csv` and they look like this
```
(base) sagemaker-user@default:/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2/classifier$ cat labels_quality.csv | head -n 5
filename,label,job_id,split
./unmasked/job_1734/h4h_ef_frames/1.2.276.0.7230010.3.1.2.845494328.1.1703598471.21532689/1.2.276.0.7230010.3.1.3.845494328.1.1703598471.21532690/mp4/1.2.276.0.7230010.3.1.4.1714578744.1.1703600527.15739402_masked.jpg,high,1734,train
./unmasked/job_1734/h4h_ef_frames/1.2.276.0.7230010.3.1.2.845494328.1.1703598471.21532689/1.2.276.0.7230010.3.1.3.845494328.1.1703598471.21532690/mp4/1.2.276.0.7230010.3.1.4.1714578744.1.1703600664.15740542_masked.jpg,med,1734,train
./unmasked/job_1734/h4h_ef_frames/1.2.276.0.7230010.3.1.2.845494328.1.1703598471.21532689/1.2.276.0.7230010.3.1.3.845494328.1.1703598471.21532690/mp4/1.2.276.0.7230010.3.1.4.811753780.1.1703598679.15799699_masked.jpg,med,1734,train
./unmasked/job_1734/h4h_ef_frames/1.2.276.0.7230010.3.1.2.845494328.1.1703598471.21532689/1.2.276.0.7230010.3.1.3.845494328.1.1703598471.21532690/mp4/1.2.276.0.7230010.3.1.4.811753780.1.1703600397.15813507_masked.jpg,high,1734,train
```

First, we map the labels to mp4:
```
python3 map_labels_to_mp4.py \
  --in labels_color.csv \
  --root /cluster/projects/bwanggroup/echo_reports/uhn_studies_22k_607/ \
  --out labels_color_mp4.csv
```

We need to convert them to the S3 paths.
```
python3 rewrite_mp4_paths_to_s3.py \
  --in labels_color_mp4.csv \
  --out labels_color_s3.csv \
  --s3-prefix s3://echodata25/results/uhn_studies_22k_607 \
  --root-marker uhn_studies_22k_607
```

Randomly sample S3 paths and ensure they exist:
```
python3 sample_and_check_s3_mp4s.py \
  --csv labels_color_s3.csv \
  --n 100 \
  --seed 0 \
  --out-prefix check100
```

Finally, put it in JEPA format
```
python3 make_view_labels_space_sep.py \
  --in labels_color_s3.csv \
  --out ../data/csv/uhn_colors_22k_train.csv \
  --mapping uhn_color_22k_mapping_train.txt \
  --split train

python3 make_view_labels_space_sep.py \
  --in labels_color_s3.csv \
  --out ../data/csv/uhn_colors_22k_test.csv \
  --mapping uhn_color_22k_mapping_test.txt \
  --split test

python3 make_view_labels_space_sep.py \
  --in labels_color_s3.csv \
  --out ../data/csv/uhn_colors_22k_val.csv \
  --mapping uhn_color_22k_mapping_val.txt \
  --split val
```

Run the training
```
python -m evals.main \
    --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/eval/vitg-384/color/color_classification_336px.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee color_336px_dec24v1.log
```


# Quality Classification
~69% accuracy
[   30] train: 87.041%  val(max-head): 69.981%

First, we map the labels to mp4:
```
python3 map_labels_to_mp4.py \
  --in labels_quality.csv \
  --root /cluster/projects/bwanggroup/echo_reports/uhn_studies_22k_607/ \
  --out labels_quality_mp4.csv
```

We need to convert them to the S3 paths.
```
python3 rewrite_mp4_paths_to_s3.py \
  --in labels_quality_mp4.csv \
  --out labels_quality_s3.csv \
  --s3-prefix s3://echodata25/results/uhn_studies_22k_607 \
  --root-marker uhn_studies_22k_607
```

Randomly sample S3 paths and ensure they exist:
```
python3 sample_and_check_s3_mp4s.py \
  --csv labels_quality_s3.csv \
  --n 100 \
  --seed 0 \
  --out-prefix check100
```

Finally, put it in JEPA format
```
python3 make_view_labels_space_sep.py \
  --in labels_quality_s3.csv \
  --out ../data/csv/uhn_quality_22k_train.csv \
  --mapping uhn_quality_22k_mapping_train.txt \
  --split train

python3 make_view_labels_space_sep.py \
  --in labels_quality_s3.csv \
  --out ../data/csv/uhn_quality_22k_test.csv \
  --mapping uhn_quality_22k_mapping_test.txt \
  --split test

python3 make_view_labels_space_sep.py \
  --in labels_quality_s3.csv \
  --out ../data/csv/uhn_quality_22k_val.csv \
  --mapping uhn_quality_22k_mapping_val.txt \
  --split val
```

Run the training
```
python -m evals.main \
    --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/eval/vitg-384/quality/quality_classification_336px.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee quality_336px_dec24v1.log
```

# Zoom Classification
~85% accuracy
[   13] train: 90.500%  val(max-head): 86.099%

First, we map the labels to mp4:
```
python3 map_labels_to_mp4.py \
  --in labels_zoom.csv \
  --root /cluster/projects/bwanggroup/echo_reports/uhn_studies_22k_607/ \
  --out labels_zoom_mp4.csv
```

We need to convert them to the S3 paths.
```
python3 rewrite_mp4_paths_to_s3.py \
  --in labels_zoom_mp4.csv \
  --out labels_zoom_s3.csv \
  --s3-prefix s3://echodata25/results/uhn_studies_22k_607 \
  --root-marker uhn_studies_22k_607
```

Randomly sample S3 paths and ensure they exist:
```
python3 sample_and_check_s3_mp4s.py \
  --csv labels_zoom_s3.csv \
  --n 100 \
  --seed 0 \
  --out-prefix check100
```

Finally, put it in JEPA format
```
python3 make_view_labels_space_sep.py \
  --in labels_zoom_s3.csv \
  --out ../data/csv/uhn_zoom_22k_train.csv \
  --mapping uhn_zoom_22k_mapping_train.txt \
  --split train

python3 make_view_labels_space_sep.py \
  --in labels_zoom_s3.csv \
  --out ../data/csv/uhn_zoom_22k_test.csv \
  --mapping uhn_zoom_22k_mapping_test.txt \
  --split test

python3 make_view_labels_space_sep.py \
  --in labels_zoom_s3.csv \
  --out ../data/csv/uhn_zoom_22k_val.csv \
  --mapping uhn_zoom_22k_mapping_val.txt \
  --split val
```

Run the training
```
mkdir -p /tmp/torch_tmp
export TMPDIR=/tmp/torch_tmp
python -m evals.main \
    --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/eval/vitg-384/zoom/zoom_classification_336px.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee zoom_336px_dec24v1.log
```

# Inference

First, fix the checkpoint to remove `module` prefixes:
```
python classifier/fix_inference_checkpoint.py \
  --input /home/sagemaker-user/user-default-efs/vjepa2/evals/vitg-384/classifier/video_classification_frozen/uhn22k-classifier-fs2-ns2-nvs1-echojepa-224px-fcl-a1g2-ep50-test/latest.pt \
  --output /home/sagemaker-user/user-default-efs/vjepa2/evals/vitg-384/classifier/video_classification_frozen/uhn22k-classifier-fs2-ns2-nvs1-echojepa-224px-fcl-a1g2-ep50-test/latest_fixed.pt
```

First, modify the config file with the `predictions_save_path` and `probe_checkpoint`

```
python index_s3.py
python append_labels.py
```

Then, run view inference (336px). Set `T` for training (val only false), `f` for focal loss:
```
./run_inf.sh \
  -t "uhn22k-classifier-fs2-ns2-nvs1-echojepa-336px-cel" \
  -c "epoch_021.pt" \
  -e 21 \
  -V "data/csv/inference_18m_vjepa_labeled.csv" \
  -B 48 \
  -H 16 \
  -b 4 \
  -r 224 \
  -d "cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7"
```


Then, run view inference (224px):
```
python -m evals.main --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/inference/vitg-384/view/uhn_22k_224px.yaml --devices cuda:0 cuda:1 cuda:2 2>&1 | tee inf_view_uhn_22k_224px.log
```

For quality
```
python -m evals.main --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/inference/vitg-384/quality/uhn_22k_336px.yaml --devices cuda:0 cuda:1 cuda:3 2>&1 | tee inf_quality_uhn_22k_336px.log
```

For zoom
```
python -m evals.main --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/inference/vitg-384/zoom/uhn_22k_224px.yaml --devices cuda:0 2>&1 | tee inf_zoom_uhn_22k_224px.log
```



Map numbers to classes:
```
python3 - <<'PY'
import json, pandas as pd

csv_in  = "/home/sagemaker-user/user-default-efs/vjepa2/predictions/uhn22k_predictions.csv"
map_in  = "/home/sagemaker-user/user-default-efs/vjepa2/classifier/uhn_views_22k_mapping_train.json"
csv_out = "/home/sagemaker-user/user-default-efs/vjepa2/predictions/uhn22k_predictions_with_names.csv"

with open(map_in, "r") as f:
    m = json.load(f)

# normalize mapping to int -> str
m = {int(k): v for k, v in m.items()}

df = pd.read_csv(csv_in)

df["true_label_name"] = df["true_label"].map(m)
df["predicted_class_name"] = df["predicted_class"].map(m)

df.to_csv(csv_out, index=False)
print("Wrote:", csv_out)
PY
```

# verify

python verify_sample.py data/csv/uhn_views_22k_train.csv
python verify_sample.py data/csv/uhn_views_22k_val.csv
python verify_sample.py data/csv/inference_18m_vjepa.csv
python verify_sample.py indices/master_index_18M.csv
python verify_sample.py data/csv/inference_18m_vjepa_labeled.csv

# clean data
python classifier/clean_dataset.py data/csv/uhn_colors_22k_train.csv data/csv/uhn_colors_22k_val.csv data/csv/uhn_colors_22k_test.csv

# color model
TRAIN_DATA="/home/sagemaker-user/user-default-efs/vjepa2/data/csv/uhn_colors_22k_train_cleaned.csv"
VAL_DATA="/home/sagemaker-user/user-default-efs/vjepa2/data/csv/uhn_colors_22k_val_cleaned.csv"
TEST_DATA="/home/sagemaker-user/user-default-efs/vjepa2/data/csv/uhn_colors_22k_test_cleaned.csv"

torchrun --nproc_per_node=8 --master_port=29501 train_convnext.py \
  --mode color \
  --train_csv "$TRAIN_DATA" \
  --val_csv "$VAL_DATA" \
  --test_csv "$TEST_DATA" \
  --batch_size 128 \
  --lr 2e-4 \
  --epochs 100 \
  --img_size 336 \
  --output_dir "./output/color_run1_convnext_small_336px"

# image model
<!-- TRAIN_DATA="/home/sagemaker-user/user-default-efs/vjepa2/data/csv/uhn_views_22k_train_cleaned.csv"
VAL_DATA="/home/sagemaker-user/user-default-efs/vjepa2/data/csv/uhn_views_22k_val_cleaned.csv"
TEST_DATA="/home/sagemaker-user/user-default-efs/vjepa2/data/csv/uhn_views_22k_test_cleaned.csv" -->

TRAIN_DATA="/home/sagemaker-user/user-default-efs/vjepa2/data/csv/uhn_views_22k_train_cleaned_336px.csv"
VAL_DATA="/home/sagemaker-user/user-default-efs/vjepa2/data/csv/uhn_views_22k_val_cleaned_336px.csv"
TEST_DATA="/home/sagemaker-user/user-default-efs/vjepa2/data/csv/uhn_views_22k_test_cleaned_336px.csv"

torchrun --nproc_per_node=8 train_convnext.py \
  --mode view \
  --train_csv "$TRAIN_DATA" \
  --val_csv "$VAL_DATA" \
  --test_csv "$TEST_DATA" \
  --batch_size 128 \
  --lr 2e-4 \
  --epochs 100 \
  --img_size 336 \
  --output_dir "./output/run5_convnext_small_336px"


# convnext base

TRAIN_DATA="/home/sagemaker-user/user-default-efs/vjepa2/data/csv/uhn_colors_22k_train_cleaned.csv"
VAL_DATA="/home/sagemaker-user/user-default-efs/vjepa2/data/csv/uhn_colors_22k_val_cleaned.csv"
TEST_DATA="/home/sagemaker-user/user-default-efs/vjepa2/data/csv/uhn_colors_22k_test_cleaned.csv"

torchrun --nproc_per_node=8 --master_port=29501 train_convnext.py \
  --mode color \
  --train_csv "$TRAIN_DATA" \
  --val_csv "$VAL_DATA" \
  --test_csv "$TEST_DATA" \
  --batch_size 50 \
  --lr 2e-4 \
  --epochs 100 \
  --img_size 336 \
  --output_dir "./output/color_run1_convnext_small_336px"

TRAIN_DATA="/home/sagemaker-user/user-default-efs/vjepa2/data/csv/uhn_views_22k_train_cleaned_336px.csv"
VAL_DATA="/home/sagemaker-user/user-default-efs/vjepa2/data/csv/uhn_views_22k_val_cleaned_336px.csv"
TEST_DATA="/home/sagemaker-user/user-default-efs/vjepa2/data/csv/uhn_views_22k_test_cleaned_336px.csv"

torchrun --nproc_per_node=8 train_convnext.py \
  --mode view \
  --train_csv "$TRAIN_DATA" \
  --val_csv "$VAL_DATA" \
  --test_csv "$TEST_DATA" \
  --batch_size 128 \
  --lr 1e-4 \
  --epochs 100 \
  --model convnext_base.fb_in1k \
  --output_dir "./output/run6_convnext_base_336px" \
  --img_size 336


# inference
torchrun --nproc_per_node=8 view_inference_18m.py \
  --input_csv "../indices/master_index_18M_cleaned.csv" \
  --output_dir "./output/view_inference_18m" \
  --num_frames 5 \
  --batch_size 128

torchrun --nproc_per_node=8 color_inference_18m.py \
  --input_csv "../indices/master_index_18M_cleaned.csv" \
  --output_dir "./output/color_inference_18m" \
  --num_frames 5 \
  --batch_size 128

# build files
cd output/color_inference_18m
awk 'FNR==1 && NR!=1{next;}{print}' color_predictions_rank*.csv > master_predictions.csv

cd output/view_inference_18m
awk 'FNR==1 && NR!=1{next;}{print}' predictions_rank*.csv > master_predictions.csv

python convert_to_parquet.py