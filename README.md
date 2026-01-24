# V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning

### Dataset
- deid_overlap_revamped.ipynb -- combined syngo and heartlab oids with deids (aws_syngo_exclusive_0806.csv, aws_heartlab_0806.csv)
- starter_labels.ipynb -- building file paths for vjepa2 classifier training (must have s3 uri and value)
- build_manifests.ipynb -- build file manifests on s3 for classification into different views (all_es_combined.parquet)
- data/build_pacemaker_dataset.ipynb -- building a4c pacemaker dataset (+ tiny subset) for fast iteration
- build_dataset.ipynb -- building labeled datasets for vjepa2 classifier training in ssv2 format
- identifiers.ipynb -- heartlab mapping to deidentified studies (patient_to_study.csv)
- heartlab_link.ipynb -- link heartlab reports, studies, and series to videos (heartlab_rep_study_video.csv)
- deid_overlap.ipynb -- overlap of syngo deid keys with data on AWS (aws_uhn.csv)

### RVSP Regression
```
python -m evals.main --fname configs/eval/vitg-384/rvsp_regression.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee multi_rvsp_0122_v1.log
```

### LVEF Regression
```
python -m evals.main \
    --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/eval/vitg-384/lvef_regression.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee lvef_regression_multi_336px_0121_v1.log
```

Inference
```
python -m evals.main --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/inference/vitg-384/lvef.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee lvef_inference_0123.log
```

Multi-level inference
```
python -m evals.main --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/inference/vitg-384/lvef_336multi.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee lvef_336multi_inference_0123.log
```

### TAPSE Regression
```
python -m evals.main \
    --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/eval/vitg-384/tapse_regression.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee tapse_regression_0107_v1.log
```



### Debug
For debugging issues.
```
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export CUDA_LAUNCH_BLOCKING=1
```

Training script may time out if S3 checkpoint upload takes too long on rank 0.
```
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
```

NCCL Error (turn off persistent workers, pin mem, reduce num workers)
```
Exception raised from recvBytes at /pytorch/torch/csrc/distributed/c10d/Utils.hpp:678 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x98 (0x7f851d9785e8 in /home/sagemaker-user/.conda/envs/vjepa2-312/lib/python3.12/site-packages/torch/lib/libc10.so)
frame #1: <unknown function> + 0x5ba8bfe (0x7f85070fabfe in /home/sagemaker-user/.conda/envs/vjepa2-312/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so)
frame #2: <unknown function> + 0x5baaf40 (0x7f85070fcf40 in /home/sagemaker-user/.conda/envs/vjepa2-312/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so)
frame #3: <unknown function> + 0x5bab84a (0x7f85070fd84a in /home/sagemaker-user/.conda/envs/vjepa2-312/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so)
frame #4: c10d::TCPStore::check(std::vector<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::allocator<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > > const&) + 0x2a9 (0x7f85070f72a9 in /home/sagemaker-user/.conda/envs/vjepa2-312/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so)
frame #5: c10d::ProcessGroupNCCL::heartbeatMonitor() + 0x379 (0x7f84c87f69f9 in /home/sagemaker-user/.conda/envs/vjepa2-312/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #6: <unknown function> + 0xd8198 (0x7f84c7501198 in /home/sagemaker-user/.conda/envs/vjepa2-312/bin/../lib/libstdc++.so.6)
frame #7: <unknown function> + 0x94ac3 (0x7f851eb8cac3 in /usr/lib/x86_64-linux-gnu/libc.so.6)
frame #8: clone + 0x44 (0x7f851ec1da04 in /usr/lib/x86_64-linux-gnu/libc.so.6)

[rank3]:[W825 07:38:02.806977494 ProcessGroupNCCL.cpp:1662] [PG ID 0 PG GUID 0(default_pg) Rank 3] Failed to check the "should dump" flag on TCPStore, (maybe TCPStore server has shut down too early), with error: failed to recv, got 0 bytes
```

### Monitor
```
chmod +x watcher.sh
./watcher.sh "/home/sagemaker-user/user-default-efs/vjepa2/checkpoints/pretrain/1.8.vitg16-336px-16f-echo-0820" 3 60
```

### Building Labels
1. `starter_labels.ipynb` -- Map the Syngo/HeartLab labels to S3 URIs (aws_uhn.csv) and simplify label types if needed.
2. `data/build_rvfx_dataset.ipynb.ipynb` -- Create train test splits and put it in VJEPA format.
3. `configs/eval/vitg-384/rvfx.yaml` -- Create your grid search.


### Inference
```
python -m evals.main --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/inference/vitg-384/rvfx.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3 2>&1 | tee rvfx_inference_0911.log
```

### Dec 21, 2025

The pretrained checkpoint path is as follows:
```
/home/sagemaker-user/user-default-efs/vjepa2/checkpoints/anneal/keep/pt-280-an81.pt
```

For a working reference script, run the following:
```
python -m evals.main \
    --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/eval/vitg-384/rvfx_bs8_ns2_anneal_f16_ssv2.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee rvfx_bs8_ns2_pt280_an80_ssv2_1221.log
```

Check opt grid
```
python3 rank_opt_grid.py \
  --exp-dir mrvsf-vitg16-336-16f-pt-e279-an-e78-fs1-ns3-nvs2 \
  --epoch 2 \
  --metric best_val_acc_per_head \
  --topk 10 \
  --print-grid-dicts
```

Grids to search
```
mrvsf-vitg16-336-16f-pt-e279-an-e78-fs1-ns3-nvs2/epoch_001.pt
mrvsf-vitg16-336-16f-pt-e279-an-e78-fs1-ns3-nvs2/epoch_002.pt
rvsf-a4c-full-vitg16-336-16f-pt-e279-an-e78-fs1-ns3-nvs2/epoch_001.pt
rvsf-a4c-full-vitg16-336-16f-pt-e279-an-e78-fs1-ns3-nvs2/latest.pt
rvsf-a4c-full-vitg16-336-16f-pt280-an80-fs1-ns2-nvs2/epoch_001.pt
rvsf-a4c-full-vitg16-336-16f-pretrain-e279-fs1-ns3-nvs2/epoch_001.pt
rvsf-a4c-full-vitg16-336-16f-pretrain-e279-fs1-ns3-nvs2/epoch_002.pt
rvsf-a4c-full-vitg16-336-16f-pretrain-e279-fs1-ns3-nvs2/epoch_003.pt
rvsf-a4c-full-vitg16-336-16f-pretrain-e279-fs1-ns3-nvs2/latest.pt
```

Run pruned grid with multilevel for classification
```
python -m evals.main \
    --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/eval/vitg-384/classification_1221_g6.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee classification_1221_v1.log
```

```
python -m evals.main \
    --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/eval/vitg-384/classification_1221_g6.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee classification_echojepa_v1.log
```

Clear extra checkpoints and keep top 3
```
python3 offload_ckpts_to_s3.py \
  --exp-dir ./classifier/video_classification_frozen/uhn22k-classifier-vitg16-336-16f-pt279-a81-fs2-ns2-nvs1 \
  --s3-prefix s3://echodata25/results/uhn22k-classifier-vitg16-336-16f-pt279-a81-fs2-ns2-nvs1/checkpoints \
  --topk 3 \
  --delete-local
```


Run experiment
```
python -m evals.main \
    --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/eval/vitg-384/classification_1221_g6_50p.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee classification_echojepa_50p_1222_v1.log
```

### Resize Datasets

Create the 224px Dataset (For Swin, EchoFM, EchoPrime)
```
export PATH="$HOME/ffmpeg_build/bin:$PATH"
python resize_dataset.py \
  --input_dir /cluster/projects/bwanggroup/echo_reports/uhn_studies_22k_585 \
  --output_dir /cluster/projects/bwanggroup/echo_reports/uhn_studies_22k_585_224px \
  --size 224 \
  --workers 32
```

Create the 112px Dataset (For PanEcho)
```
export PATH="$HOME/ffmpeg_build/bin:$PATH"
python resize_dataset.py \
  --input_dir /cluster/projects/bwanggroup/echo_reports/uhn_studies_22k_585 \
  --output_dir /cluster/projects/bwanggroup/echo_reports/uhn_studies_22k_585_112px \
  --size 112 \
  --workers 32
```

### Classifier Training

Download checkpoint
```
cd /home/sagemaker-user/user-default-efs/vjepa2/checkpoints/pretrain/keep
aws s3 cp s3://echodata25/vjepa2/checkpoints-0820/e279.pt .
```

```
cd /home/sagemaker-user/user-default-efs/vjepa2/checkpoints/anneal/keep
aws s3 cp s3://echodata25/vjepa2/anneal-0828/e39.pt .
```

RVFX 81ep Anneal:
```
python -m evals.main --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/eval/vitg-384/rvfx_bs8_ns2_anneal_f16_ssv2.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee rvfx_bs8_ns2_pt280_an80_ssv2_1221.log
```

Small Exp:
```
unset SLURM_LOCALID
CUDA_VISIBLE_DEVICES=0 python -m evals.main \
    --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/eval/vitg-384/rvfx_bs8_ns2_anneal_ssv2_min.yaml \
    --devices cuda:0 2>&1 | tee rvfx_bs8_ns2_anneal_e39_ssv2_min_0919.log

unset SLURM_LOCALID  
CUDA_VISIBLE_DEVICES=3 python -m evals.main \
  --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/eval/vitg-384/rvfx_bs8_ns2_pretrain_ssv2_min.yaml \
  --devices cuda:3 2>&1 | tee rvfx_bs8_ns2_pretrain_e249_ssv2_min_0919.log
```
```
python -m evals.main --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/eval/vitg-384/rvfx_bs8_ns2_anneal_ssv2_min.yaml --devices cuda:0 2>&1 | tee rvfx_bs8_ns2_anneal_e39_ssv2_min_0919.log
python -m evals.main --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/eval/vitg-384/rvfx_bs8_ns2_pretrain_ssv2_min.yaml --devices cuda:1 2>&1 | tee rvfx_bs8_ns2_pretrain_e198_ssv2_min_0919.log
```


RVFX: H32, B6
```
python -m evals.main --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/eval/vitg-384/rvfx_h32_b6.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee rvfx_h32_b6_0831.log
```

LAD: 39ep anneal
```
python -m evals.main --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/eval/vitg-384/lad_bs8_ns2_anneal_ssv2.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee lad_bs8_ns2_anneal39_0922_moderate_v0.log
```

TVR: 39ep anneal
```
python -m evals.main --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/eval/vitg-384/tvr_bs8_ns2_anneal_ssv2.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee tvr_bs8_ns2_anneal39_0912_v0.log
```

MVR: 39ep anneal
```
python -m evals.main --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/eval/vitg-384/mvr_bs8_ns2_anneal_ssv2.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee mvr_bs8_ns2_anneal39_0911_v2.log
```

RVFX: Final pretrain, 39ep anneal
```
python -m evals.main --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/eval/vitg-384/rvfx_bs8_ns2_pt.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee rvfx_bs8_ns2_pt200_0830.log

python -m evals.main --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/eval/vitg-384/rvfx_bs8_ns2_anneal.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee rvfx_bs8_ns2_anneal39_0905.log


python -m evals.main --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/eval/vitg-384/rvfx_bs8_ns2_anneal_ssv2.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee rvfx_bs8_ns2_anneal39_ssv2_0905.log

python -m evals.main --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/eval/vitg-384/rvfx_bs8_ns2_pretrain_ssv2.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee rvfx_bs8_ns2_pretrain279_ssv2_0920.log

export TMPDIR=/dev/shm
export TEMP=/dev/shm
export TMP=/dev/shm
export PYTORCH_SHARING_STRATEGY=file_descriptor

python -m evals.main --fname configs/eval/vitg-384/rvfx_bs8_ns2_anneal_f16_ssv2.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee rvfx_bs8_ns2_pt279_an79_1004.log

export TMPDIR=/dev/shm
export TEMP=/dev/shm
export TMP=/dev/shm
export PYTORCH_SHARING_STRATEGY=file_descriptor

python -m evals.main --fname configs/eval/vitg-384/multi_rvfx_bs8_ns2_anneal_f16_ssv2.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee multi_rvfx_bs8_ns2_pt279_an79_1005_v1.log
```

Best settings for RVFX
```
python -m evals.main --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/eval/vitg-384/rvfx_bs8_ns2.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee rvfx_bs8_ns2_0828.log
```

Back to small batch size
```
python -m evals.main --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/eval/vitg-384/rvfx_cooldown_v3.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee rvfx_cooldown_h16_b4_bs8_ep162_0828_fs2_ns2_FULL.log

python -m evals.main --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/eval/vitg-384/rvfx_cooldown_v2.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee rvfx_cooldown_h44_b4_bs8_ep162_0828_fs2_ns2.log
```

New configs (bs48, scaled LR #2)
```
python -m evals.main --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/eval/vitg-384/rvfx_pretrain_v2.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee rvfx_kinetics_h16_b4_bs48_ep144_0827_scaledLR_v2.log

python -m evals.main --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/eval/vitg-384/rvfx_pretrain_v3.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee rvfx_kinetics_h16_b4_bs48_ep144_0827_scaledLR_v3.log
```

Full A4C dataset, RV systolic function
```
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m evals.main --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/eval/vitg-384/rvfx_full.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee rvfx_kinetics_h16_b4_bs48_ep144_0826_FULL_scaledLR.log
```

Scaled (higher batch size, 8 to 48) RV systolic function
```
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m evals.main --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/eval/vitg-384/rvfx_pretrain.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee rvfx_kinetics_h16_b4_bs48_ep144_0826_scaledLR.log
```

RV systolic function (remember to modify **checkpoint**, **run tag**, **output file**)
```
python -m evals.main --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/eval/vitg-384/rvfx_kinetics.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee rvfx_kinetics_h32_b8_v1.log

python -m evals.main --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/eval/vitg-384/rvfx_cooldown.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee rvfx_cooldown_h16_b4_0824_keepe96_b8.log
```

Pacemaker detection
```
python -m evals.main --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/eval/vitg-384/pacemaker.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee pacemaker_v1.log
```

Mitral valve regurgitation (old)
```
python -m evals.main --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/eval/vitg-384/echo_mvr.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
```

Sample outputs. `[iteration num]` `[max acc]` `[mean min]` (across all heads).
```
[INFO][2025-07-23 12:10:29][root][run_one_epoch] [0] 53.125% [20.156% 3.125%] [mem: 5.09e+04]
[INFO][2025-07-23 12:12:51][root][run_one_epoch] [10] 46.875% [33.026% 19.602%] [mem: 6.51e+04]
```

### Run Annealing
Cooldown run.
> Note: Make sure you create a brand new folder for the run and set force_load_pretrain to true from your final pretrain checkpoint.
```
python -m app.main --fname configs/train/vitg16/cooldown-echo-336px-16f-0930.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee cooldown-echo-336px-16f-ep279-0930_v1.log
```

### Run SSL Pretraining

Set environment guards
```
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_DISABLE_SIGNAL_HANDLERS=1   # quieter shutdown
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
```

(New) cooldown script with LR adjusted to global batch and token ratios.
```
python -m app.main --fname configs/train/vitg16/pretrain-echo-336px-16f-0820.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee pretrain-echo-336px-16f-ep200-400-0922.log
```

(Old) Run pretraining with domain and LR adaptation (better if training from scratch).
```
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,expandable_segments:True"
python -m app.main --fname configs/train/vitg16/pretrain-336px-16f-echo.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
```

(Old) cooldown parameters will barely move the weights.
```
python -m app.main --fname configs/train/vitg16/cooldown-336px-64f.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
```


### Data Curation

1. `create_annotations.py` -- Create a pretraining dataset in the format expected by VJEPA2.
2. `monitor_checkpoints.py` -- Keep track of checkpoint directory and upload periodically to S3. Delete extra checkpoints as needed.
3. `build_manifests.ipynb` -- Build file manifests. Combine dataframes (es, es1, es2). Check MP4 paths. Fix MP4 paths. Filter for A4C. Connect A4C videos to labels (mitral valve regurgitation).
4. `batch_classify.py` -- Classify all images in file manifests into canonical echo views.
5. `scratch.ipynb` -- Create mitral regurgitation data splits. Create loss and accuracy curves.


### Modifications

Run echo classification pipeline.
```
OUT_DIR=./class_preds_es0 torchrun --nproc_per_node=8 batch_classify.py   --bucket echodata25   --manifest_s3 s3://echodata25/results/echo-images/all_unmasked_png_paths_0_v2.clean.txt.gz   --model_s3 s3://echodata25/results/models/view_classifier/best_f1_84.pt   --batch_size 2048

OUT_DIR=./class_preds_es1 torchrun --nproc_per_node=8 batch_classify.py   --bucket echodata25   --manifest_s3 s3://echodata25/results/echo-images/all_unmasked_png_paths_1_v2.clean.txt.gz   --model_s3 s3://echodata25/results/models/view_classifier/best_f1_84.pt   --batch_size 2048

OUT_DIR=./class_preds_es2 torchrun --nproc_per_node=8 batch_classify.py   --bucket echodata25   --manifest_s3 s3://echodata25/results/echo-images/all_unmasked_png_paths_2_v2.clean.dedup.txt.gz   --model_s3 s3://echodata25/results/models/view_classifier/best_f1_84.pt   --batch_size 2048
```

Run echo classification pipeline (again).
```
MAX_WORKERS=512 OUT_DIR=./class_preds_es0 torchrun --nproc_per_node=8 batch_classify.py   --bucket echodata25   --manifest_s3 s3://echodata25/results/echo-images/all_unmasked_png_paths_0_v2_rem.clean.txt.gz   --model_s3 s3://echodata25/results/models/view_classifier/best_f1_84.pt   --batch_size 16384

MAX_WORKERS=512 OUT_DIR=./class_preds_es1 torchrun --nproc_per_node=8 batch_classify.py   --bucket echodata25   --manifest_s3 s3://echodata25/results/echo-images/all_unmasked_png_paths_1_v2_rem.clean.txt.gz   --model_s3 s3://echodata25/results/models/view_classifier/best_f1_84.pt   --batch_size 16384

MAX_WORKERS=512 OUT_DIR=./class_preds_es2 torchrun --nproc_per_node=8 batch_classify.py   --bucket echodata25   --manifest_s3 s3://echodata25/results/echo-images/all_unmasked_png_paths_2_v2_rem.clean.txt.gz   --model_s3 s3://echodata25/results/models/view_classifier/best_f1_84.pt   --batch_size 16384
```


### [Meta FAIR](https://ai.meta.com/research/)

Mahmoud Assran∗, Adrien Bardes∗, David Fan∗, Quentin Garrido∗, Russell Howes∗, Mojtaba
Komeili∗, Matthew Muckley∗, Ammar Rizvi∗, Claire Roberts∗, Koustuv Sinha∗, Artem Zholus*,
Sergio Arnaud*, Abha Gejji*, Ada Martin*, Francois Robert Hogan*, Daniel Dugas*, Piotr
Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil
Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier*, Yann LeCun*, Michael
Rabbat*, Nicolas Ballas*

*Core Team

[[`Paper`](https://arxiv.org/abs/2506.09985)] [[`Blog`](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks)] [[`BibTex`](#Citation)]

Official Pytorch codebase for V-JEPA 2 and V-JEPA 2-AC.

V-JEPA 2 is a self-supervised approach to training video encoders, using internet-scale video data, that attains state-of-the-art performance on motion understanding and human action anticpation tasks. V-JEPA 2-AC is a latent action-conditioned world model post-trained from V-JEPA 2 (using a small amount of robot trajectory interaction data) that solves robot manipulation tasks without environment-specific data collection or task-specific training or calibration.

<p align="center">
	<img src="assets/flowchart.png" width=100%>
</p>

<!---
## Updates

* **[Jun-6-25]:** V-JEPA 2 is released. [[`Blog`](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks)]
--->

## V-JEPA 2 Pre-training

**(Top)** The encoder and predictor are pre-trained through self-supervised learning from video using a masked latent feature prediction objective, leveraging abundant natural videos to bootstrap physical world understanding and prediction. **(Bottom)** Performance of V-JEPA 2 on downstream understanding and prediction tasks.

<img align="left" src="https://dl.fbaipublicfiles.com/vjepa2/vjepa2-pretrain.gif" width=65%>&nbsp;
<table>
  <tr>
    <th colspan="1">Benchmark</th>
    <th colspan="1">VJEPA 2</th>
    <th colspan="1">Previous Best</th>
  </tr>
  <tr>
    <td>EK100</td>
    <td>39.7%</td>
    <td>27.6% (PlausiVL)</td>
  </tr>
  <tr>
    <td>SSv2 (Probe)</td>
    <td>77.3%</td>
    <td>69.7% (InternVideo2-1B)</td>
  </tr>
  <tr>
    <td>Diving48 (Probe)</td>
    <td>90.2%</td>
    <td>86.4% (InternVideo2-1B)</td>
  </tr>
  <tr>
    <td>MVP (Video QA)</td>
    <td>44.5%</td>
    <td>39.9% (InternVL-2.5)</td>
  </tr>
  <tr>
    <td>TempCompass (Video QA)</td>
    <td>76.9%</td>
    <td>75.3% (Tarsier 2)</td>
  </tr>
</table>

## V-JEPA 2-AC Post-training

**(Top)** After post-training with a small amount of robot data, we can deploy the model on a robot arm in new environments, and tackle foundational tasks like reaching, grasping, and pick-and-place by planning from image goals. **(Bottom)** Performance on robot maniuplation tasks using a Franka arm, with input provided through a monocular RGB camera.

<img align="left" src="https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-planning.gif" width=65%>&nbsp;
<table>
  <tr>
    <th colspan="1"></th>
    <th colspan="1"></th>
    <th colspan="2">Grasp</th>
    <th colspan="2">Pick-and-Place</th>
  </tr>
  <tr>
    <th colspan="1">Method</th>
    <th colspan="1">Reach</th>
    <th colspan="1">Cup</th>
    <th colspan="1">Box</th>
    <th colspan="1">Cup</th>
    <th colspan="1">Box</th>
  </tr>
  <tr>
    <td>Octo</td>
    <td>100%</td>
    <td>10%</td>
    <td>0%</td>
    <td>10%</td>
    <td>10%</td>
  </tr>
  <tr>
    <td>Cosmos</td>
    <td>80%</td>
    <td>0%</td>
    <td>20%</td>
    <td>0%</td>
    <td>0%</td>
  </tr>
  <tr>
    <td>VJEPA 2-AC</td>
    <td>100%</td>
    <td>60%</td>
    <td>20%</td>
    <td>80%</td>
    <td>50%</td>
  </tr>
</table>

## Models

### V-JEPA 2

#### HuggingFace

See our [HuggingFace collection](https://huggingface.co/collections/facebook/v-jepa-2-6841bad8413014e185b497a6) for V-JEPA 2.

#### Pretrained Checkpoints

<table>
  <tr>
    <th colspan="1">Model</th>
    <th colspan="1">#Parameters</th>
    <th colspan="1">Resolution</th>
    <th colspan="1">Download Link</th>
    <th colspan="1">Pretraining Config</th>
  </tr>
  <tr>
    <td>ViT-L/16</td>
    <td>300M</td>
    <td>256</td>
    <td><a href="https://dl.fbaipublicfiles.com/vjepa2/vitl.pt">checkpoint</a></td>
    <td><a href="configs/train/vitl16">configs</a></td>
  </tr>
  <tr>
    <td>ViT-H/16</td>
    <td>600M</td>
    <td>256</td>
    <td><a href="https://dl.fbaipublicfiles.com/vjepa2/vith.pt">checkpoint</a></td>
    <td><a href="configs/train/vith16/">configs</a></td>
  </tr>
  <tr>
    <td>ViT-g/16</td>
    <td>1B</td>
    <td>256</td>
    <td><a href="https://dl.fbaipublicfiles.com/vjepa2/vitg.pt">checkpoint</a></td>
    <td><a href="configs/train/vitg16">configs</a></td>
  </tr>
  <tr>
    <td>ViT-g/16<sub>384</sub></td>
    <td>1B</td>
    <td>384</td>
    <td><a href="https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt">checkpoint</a></td>
    <td><a href="configs/train/vitg16">configs</a></td>
  </tr>
</table>

#### Pretrained backbones (via PyTorch Hub)

Please install [Pytorch](https://pytorch.org/get-started/locally/), [timm](https://pypi.org/project/timm/) and [einops](https://pypi.org/project/einops/) locally, then run the following to load each model. Installing Pytorch with CUDA support is strongly recommended.

```python
import torch

# preprocessor
processor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_preprocessor')
# models
vjepa2_vit_large = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_large')
vjepa2_vit_huge = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_huge')
vjepa2_vit_giant = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_giant')
vjepa2_vit_giant_384 = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_giant_384')

```

#### Pretrained checkpoints on Huggingface

You can also use our pretrained checkpoints on [Huggingface](https://huggingface.co/collections/facebook/v-jepa-2-6841bad8413014e185b497a6).

```python
from transformers import AutoVideoProcessor, AutoModel

hf_repo = "facebook/vjepa2-vitg-fpc64-256"
# facebook/vjepa2-vitl-fpc64-256
# facebook/vjepa2-vith-fpc64-256
# facebook/vjepa2-vitg-fpc64-256
# facebook/vjepa2-vitg-fpc64-384


model = AutoModel.from_pretrained(hf_repo)
processor = AutoVideoProcessor.from_pretrained(hf_repo)
```

#### Evaluation Attentive Probes

We share the trained attentive probes for two of our visual understanding evals (Something-Something v2 and Diving48) and the action anticipation eval EPIC-KITCHENS-100.

<table>
  <tr>
    <th colspan="1">Model</th>
    <th colspan="4">SSv2</th>
    <th colspan="4">Diving48</th>
    <th colspan="4">EK100</th>
  </tr>
  <tr>
    <th colspan="1"></th>
    <th colspan="1">Checkpoint</th>
    <th colspan="1">Training Config</th>
    <th colspan="1">Inference Config</th>
    <th colspan="1">Result</th>
    <th colspan="1">Checkpoint</th>
    <th colspan="1">Training Config</th>
    <th colspan="1">Inference Config</th>
    <th colspan="1">Result</th>
    <th colspan="1">Checkpoint</th>
    <th colspan="1">Training Config</th>
    <th colspan="1">Inference Config</th>
    <th colspan="1">Result</th>
  </tr>
  <tr>
    <td>ViT-L/16</td>
    <td><a href="https://dl.fbaipublicfiles.com/vjepa2/evals/ssv2-vitl-16x2x3.pt">checkpoint</a></td>
    <td><a href="configs/eval/vitl/ssv2.yaml">config</a></td>
    <td><a href="configs/inference/vitl/ssv2.yaml">config</a></td>
    <td>73.7%</td>
    <td><a href="https://dl.fbaipublicfiles.com/vjepa2/evals/diving48-vitl-256.pt">checkpoint</a></td>
    <td><a href="configs/eval/vitl/diving48.yaml">config</a></td>
    <td><a href="configs/inference/vitl/diving48.yaml">config</a></td>
    <td>89.0%</td>
    <td><a href="https://dl.fbaipublicfiles.com/vjepa2/evals/ek100-vitl-256.pt">checkpoint</a></td>
    <td><a href="configs/eval/vitl/ek100.yaml">config</a></td>
    <td><a href="configs/inference/vitl/ek100.yaml">config</a></td>
    <td>32.7 R@5</td>
  </tr>
  <tr>
    <td>ViT-g/16<sub>384</td>
    <td><a href="https://dl.fbaipublicfiles.com/vjepa2/evals/ssv2-vitg-384-64x2x3.pt">checkpoint</a></td>
    <td><a href="configs/eval/vitg-384/ssv2.yaml">config</a></td>
    <td><a href="configs/inference/vitg-384/ssv2.yaml">config</a></td>
    <td>77.3%</td>
    <td><a href="https://dl.fbaipublicfiles.com/vjepa2/evals/diving48-vitg-384-32x4x3.pt">checkpoint</a></td>
    <td><a href="configs/eval/vitg-384/diving48.yaml">config</a></td>
    <td><a href="configs/inference/vitg-384/diving48.yaml">config</a></td>
    <td>90.2%</td>
    <td><a href="https://dl.fbaipublicfiles.com/vjepa2/evals/ek100-vitg-384.pt">checkpoint</a></td>
    <td><a href="configs/eval/vitg-384/ek100.yaml">config</a></td>
    <td><a href="configs/inference/vitg-384/ek100.yaml">config</a></td>
    <td>39.7 R@5</td>
  </tr>
</table>

### V-JEPA 2-AC

Our action-conditioned checkpoint was trained from the ViT-g encoder.
<table>
  <tr>
    <th colspan="1">Model</th>
    <th colspan="1">Download Link</th>
    <th colspan="1">Training Config</th>
  </tr>
  <tr>
    <td>ViT-g/16</td>
    <td><a href="https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt">checkpoint</a></td>
    <td><a href="configs/train/vitg16/droid-256px-8f.yaml">config</a></td>
  </tr>
</table>

#### Pretrained action-conditioned backbone (via PyTorch Hub)

Please install [Pytorch](https://pytorch.org/get-started/locally/), [timm](https://pypi.org/project/timm/) and [einops](https://pypi.org/project/einops/) locally, then run the following to load each model. Installing Pytorch with CUDA support is strongly recommended.

```python
import torch

vjepa2_encoder, vjepa2_ac_predictor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_ac_vit_giant')
```

See [energy_landscape_example.ipynb](notebooks/energy_landscape_example.ipynb) for an example notebook computing the energy landscape of the pretrained action-conditioned backbone using a robot trajectory collected from our lab.
To run this notebook, you'll need to aditionally install [Jupyter](https://jupyter.org/install) and [Scipy](https://scipy.org/install/) in your conda environment.

## Getting Started

### Setup

```
conda create -n vjepa2-312 python=3.12
conda activate vjepa2-312
pip install .  # or `pip install -e .` for development mode
```

### Usage Demo

See [vjepa2_demo.ipynb](notebooks/vjepa2_demo.ipynb) [(Colab Link)](https://colab.research.google.com/github/facebookresearch/vjepa2/blob/main/notebooks/vjepa2_demo.ipynb) or [vjepa2_demo.py](notebooks/vjepa2_demo.py) for an example of how to load both the HuggingFace and PyTorch V-JEPA 2 models and run inference on a sample video to get a sample classification result.

The script assumes the presence of downloaded model checkpoints so you will need to download the model weights and update the corresponding paths in the script. E.g.:
```
wget https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt -P YOUR_DIR
wget https://dl.fbaipublicfiles.com/vjepa2/evals/ssv2-vitg-384-64x2x3.pt -P YOUR_DIR

# Then update your model paths in vjepa2_demo.py.
pt_model_path = YOUR_DIR/vitg-384.pt
classifier_model_path = YOUR_DIR/ssv2-vitg-384-64x2x3.pt

# Then run the script (assumes your machine has a GPU)
python -m notebooks.vjepa2_demo
```

### Probe-based evaluation

Probe-based evaluation consists in training an attentive probe on top of frozen V-JEPA 2 features. We provide training scripts for training your own probes, and checkpoints to run inference directly.

#### Training probes

Evaluations can be run either locally, or distributed via SLURM. (Running locally is useful for debugging and validation).
These sample commands launch Something-Something v2 video classification; other evals are launched by specifying the corresponding config.
Use provided training configs under "Evaluation Attentive Probes". These configs allow to train multiple probes in parrallel with various optimization parameters.
Change filepaths as needed (e.g. `folder`, `checkpoint`, `dataset_train`, `dataset_val`) to match locations of data and downloaded checkpoints on your local filesystem.
Change \# nodes and local batch size as needed to not exceed available GPU memory.

##### Local

To run locally, specify the GPUs to use on
```
python -m evals.main --fname configs/eval/vitl16/ssv2.yaml \
  --devices cuda:0 cuda:1
```

##### Distributed

```
python -m evals.main_distributed \
  --fname configs/eval/vitl/ssv2.yaml  \
  --time 8600 \
  --account my_account --qos=my_qos
```

#### Inference from existing probes

Use provided inference configs under [Evaluation Attentive Probes](#evaluation-attentive-probes).
Download the corresponding checkpoint, rename it to 'latest.pt', and create a folder with the checkpoint inside, with the format matching the variables in the config:
```
[folder]/[eval_name]/[tag]/latest.pt
```
Then run inference, locally or distributed, using the same evaluation commands as above, but with configs from `configs/inference`.

### Pretraining

Likewise, training can also be run locally or distributed. Pretraining and cooldown training phases are
run with the same command using different configs.
These sample commands launch initial training of a ViT-L model. Configs for cooldown (or action-conditioned) training
can be found in the same directory as the config for initial training.

#### Local

```
python -m app.main --fname configs/train/vitl16/pretrain-256px-16f.yaml \
  --devices cuda:0
```

#### Distributed

```
python -m app.main_distributed \
  --fname configs/train/vitl16/pretrain-256px-16f.yaml
  --time 6000
  --account my_account --qos=my_qos
```

### Postraining

Post-training of the action-conditioned model, starting from the pretrained VJEPA 2 backbone, also follows a similar interface, and can be run locally or distributed using [this config](configs/train/vitg16/droid-256px-8f.yaml).
We post-train the model starting from the ViT-g/16 backbone.

#### Local

```
python -m app.main --fname configs/train/vitg16/droid-256px-8f.yaml \
  --devices cuda:0
```

#### Distributed

```
python -m app.main_distributed \
  --fname configs/train/vitg16/droid-256px-8f.yaml
  --time 6000
  --account my_account --qos=my_qos
```


## Code Structure

```
.
├── app                              # training loops
│   ├── vjepa                        #   video JEPA pre-training
│   ├── vjepa_droid                  #   training the action-conditioned model
│   ├── main_distributed.py          #   entrypoint for launch app on slurm cluster
│   └── main.py                      #   entrypoint for launch app locally on your machine
├── configs                          # config files with experiment params for training and evaluation
│   ├── train                        #   pretraining (phase 1), cooldown (phase 2), and action-conditioned training
│   └── eval                         #   frozen evaluations
├── evals                            # evaluation loops training an attentive probe with frozen backbone...
│   ├── action_anticipation_frozen   #   action anticipation
│   ├── image_classification_frozen  #   image understanding
│   ├── video_classification_frozen  #   video understanding
│   ├── main_distributed.py          #   entrypoint for distributed evaluations
│   └── main.py                      #   entrypoint for locally-run evaluations
├── src                              # the package
│   ├── datasets                     #   datasets, data loaders, ...
│   ├── models                       #   model definitions
│   ├── masks                        #   mask collators, masking utilities, ...
│   └── utils                        #   shared utilities
├── tests                            # unit tests for some modules in `src`

```

## License

The majority of V-JEPA 2 is licensed under MIT, however portions of the project are available under separate license terms:

[src/datasets/utils/video/randaugment.py](src/datasets/utils/video/randaugment.py)<br>
[src/datasets/utils/video/randerase.py](src/datasets/utils/video/randerase.py)<br>
[src/datasets/utils/worker_init_fn.py](src/datasets/utils/worker_init_fn.py)<br>

are licensed under the Apache 2.0 license.


## Citation
If you find this repository useful in your research, please consider giving a star :star: and a citation
```bibtex
@article{assran2025vjepa2,
  title={V-JEPA~2: Self-Supervised Video Models Enable Understanding, Prediction and Planning},
  author={Assran, Mahmoud and Bardes, Adrien and Fan, David and Garrido, Quentin and Howes, Russell and
Komeili, Mojtaba and Muckley, Matthew and Rizvi, Ammar and Roberts, Claire and Sinha, Koustuv and Zholus, Artem and
Arnaud, Sergio and Gejji, Abha and Martin, Ada and Robert Hogan, Francois and Dugas, Daniel and
Bojanowski, Piotr and Khalidov, Vasil and Labatut, Patrick and Massa, Francisco and Szafraniec, Marc and
Krishnakumar, Kapil and Li, Yong and Ma, Xiaodong and Chandar, Sarath and Meier, Franziska and LeCun, Yann and
Rabbat, Michael and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2506.09985},
  year={2025}
}
```
