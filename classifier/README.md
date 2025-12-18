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


# Multi-Frame (Requires Videos)

Compile a list of studies to download (585)
```
python make_selected_study_dirs.py \
  --labels labels_masked_inplace.csv \
  --es es.txt \
  --es1 es1.txt \
  --es2 es2.txt \
  --out selected_study_dirs.csv
```

Download the studies from S3
```
python download_selected_studies.py \
  --csv selected_study_dirs.csv \
  --dest "/Users/alifmunim/Documents/Classification" \
  --log download.log \
  --only-show-errors
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

To map to the actual mp4 directories, use this script
```
python3 map_labels_to_mp4.py \
  --in labels_patient_split.csv \
  --root /cluster/projects/bwanggroup/echo_reports/uhn_studies_22k_585/ \
  --out labels_patient_split_mp4.csv \
  --check-exists \
  --drop-missing
```

Now, we have a dataset file with valid splits ready to train!




