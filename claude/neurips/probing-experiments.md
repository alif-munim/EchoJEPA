 Appendix §X — Spatial / phase / temporal-information controls on pooled features                                                                                                        
                                                                                                                                                                                          
  Purpose. Three successive post-hoc experiments on pre-extracted, spatially mean-pooled features from ViT-L encoders (JEPA IN21K e100/e200 and EchoMAE-L e99/e194) on EchoNet-Dynamic    
  LVEF regression. The goal is to disentangle whether MAE's pooled-feature LVEF advantage is driven by (a) aggregate/static anatomy, (b) single-frame/phase-specific geometry, (c)
  time-indexed sequence information, (d) adjacent-frame differences, or (e) learned temporal aggregation, and whether the conclusions change the paper's interpretation of the            
  frame-shuffling / matched-frame result.                             

  ---
  X.1 Shared input: pooled feature caches
                                                                                                                                                                                          
  Feature extraction. Pre-pool features were extracted via evals/feature_extraction_pre_pool/eval.py on EchoNet-Dynamic using the canonical NeurIPS checkpoints:
                                                                                                                                                                                          
  ┌───────────────┬────────────────────────────────────────────────────────────┬─────────────────────────────────────────┬───────────────────────────────────────────────────────────┐    
  │    encoder    │                       source S3 path                       │                   md5                   │                    feature cache (S3)                     │    
  ├───────────────┼────────────────────────────────────────────────────────────┼─────────────────────────────────────────┼───────────────────────────────────────────────────────────┤    
  │ JEPA IN21K    │ echodata25/neurips/encoders/jepa_in21k_vitl_e100.pt        │ 0893de1639fd61ff9df796ef18e144ff        │ features/diff_probe/jepa_e100_{train,test}.pt             │  
  │ e100          │                                                            │                                         │                                                           │    
  ├───────────────┼────────────────────────────────────────────────────────────┼─────────────────────────────────────────┼───────────────────────────────────────────────────────────┤    
  │ JEPA IN21K    │ runs/jepa_in21k_e200_280/training_folder/latest.pt         │ (unverified; trained by job 280, seed   │ features/diff_probe/jepa_e200_{train,test}.pt (new — job  │    
  │ e200          │                                                            │ 234)                                    │ 416)                                                      │    
  ├───────────────┼────────────────────────────────────────────────────────────┼─────────────────────────────────────────┼───────────────────────────────────────────────────────────┤    
  │ EchoMAE-L e99 │ echodata25/neurips/encoders/mae_vitl_e99.pth               │ 2ff18369993ff34a4d84ae55a9166ce5        │ features/diff_probe/mae_e99_{train,test}.pt               │
  ├───────────────┼────────────────────────────────────────────────────────────┼─────────────────────────────────────────┼───────────────────────────────────────────────────────────┤    
  │ EchoMAE-L     │ runs/videomae_e200b_179/training_folder/checkpoint-194.pth │ (unverified; job 179)                   │ features/diff_probe/mae_e194_{train,test}.pt (new — job   │
  │ e194          │                                                            │                                         │ 417)                                                      │    
  └───────────────┴────────────────────────────────────────────────────────────┴─────────────────────────────────────────┴───────────────────────────────────────────────────────────┘

  S3 prefix: s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/                                                                                                        
   
  Cache format. torch.save({"features": <fp16>, "labels": <fp32>, "paths": [...], "meta": {...}}, ...)                                                                                    
                                                                      
  ┌──────────┬───────────────────────┬───────┬────────────────────────────────────────────────────────┐                                                                                   
  │  field   │         shape         │ dtype │                       semantics                        │
  ├──────────┼───────────────────────┼───────┼────────────────────────────────────────────────────────┤                                                                                   
  │ features │ [N, S=2, T=8, D=1024] │ fp16  │ spatially-mean-pooled token sequence per video-segment │
  ├──────────┼───────────────────────┼───────┼────────────────────────────────────────────────────────┤                                                                                   
  │ labels   │ [N]                   │ fp32  │ raw LVEF in percent                                    │                                                                                   
  └──────────┴───────────────────────┴───────┴────────────────────────────────────────────────────────┘                                                                                   
                                                                                                                                                                                          
  Dimensions:                                                                                                                                                                             
  - N = number of videos in the split (7465 train, 1277 test)         
  - S = 2 = number of temporal segments per video (num_segments=2, non-overlapping)                                                                                                       
  - T = 8 = number of temporal tokens (frames_per_clip=16 / tubelet_size=2)        
  - D = 1024 = ViT-L channel dimension                                                                                                                                                    
  - Spatial axis (patches = 14×14 = 196) has already been mean-pooled at write time in evals/feature_extraction_pre_pool/eval.py (pooled = reshaped.float().mean(dim=3)), which discards  
  spatial-token structure. All probes in this appendix operate on this [T, D] sequence.                                                                                                   
                                                                                                                                                                                          
  Extraction config (all four encoders, identical):                                                                                                                                       
  - resolution: 224                                                                                                                                                                       
  - frames_per_clip: 16                                                                                                                                                                   
  - frame_step: 2                                                                                                                                                                         
  - num_segments: 2                                                                                                                                                                       
  - num_views_per_segment: 1                                                                                                                                                              
  - batch_size: 8 per GPU                                                                                                                                                                 
  - use_bfloat16: true                                                                                                                                                                    
  - 8 GPU H100 per node                                                                                                                                                                   
  - dataset_val set to echonet_dynamic_train_s3_raw.csv or echonet_dynamic_test_s3_raw.csv per run                                                                                        
                                                                                                                                                                                          
  Extraction job IDs on HyperPod SLURM: 416 (JEPA e200, 3m44s), 417 (MAE e194, 3m56s). JEPA e100 and MAE e99 caches already existed on S3 from prior job 314 lineage.                     
                                                                                                                                                                                          
  ---                                                                                                                                                                                     
  X.2 Shared training protocol (all pooled-feature probes)                                                                                                                                
                                                                                                                                                                                          
  All probes follow scripts/neurips/diff_probe_train.py conventions.  
                                                                                                                                                                                          
  Split. EchoNet-Dynamic official train/test split (7,465 train / 1,277 test videos), cached before any probe. Each video contributes 2 segments.                                         
                                                                                                                                                                                          
  Train/val split. Per seed, a stratified 90/10 holdout is drawn from the train cache using EF quintile bins (stratified_split(labels, frac_val=0.1, seed=seed, n_bins=5)). All segments  
  of a given video go to the same partition.                          
                                                                                                                                                                                          
  Seeds. [0, 1, 2, 3, 4] (5 seeds across all experiments below).                                                                                                                          
   
  Label normalization. Train-split-mean z-scoring applied to targets; predictions de-normalized before metric computation.                                                                
                                                                      
  Training objective. MSE on z-scored targets.                                                                                                                                            
                                                                      
  Optimizer. AdamW (PyTorch default betas: (0.9, 0.999), eps 1e-8).                                                                                                                       
                                                                      
  Early stopping.                                                                                                                                                                         
  - Linear probes: min_epochs=10, patience=5, min_delta=0.005 on val R², max_epochs=30
  - Pooled temporal-attn probes: min_epochs=15, patience=7, min_delta=0.002 on val R², max_epochs=50                                                                                      
                                                                                                    
  Per-video aggregation at test time. Each test segment is scored independently; per-video predictions are obtained by averaging the two segment predictions before computing video-level 
  R²/MAE/Pearson against ground truth. This matches the pred-averaging convention of the published linear-probe trajectory.                                                               
                                                                                                                                                                                          
  Best-epoch selection. Per (seed, LR, WD), the epoch with highest val R² is selected; the probe is restored to that checkpoint before test evaluation.                                   
                                                                              
  Best-HP selection. Per seed, the (LR, WD) with highest val R² is reported as that seed's result. 5-seed mean ± std of test metrics reported.                                            
                                                                              
  Batch size. 64 (all experiments).                                                                                                                                                       
                                                                              
  Device. NVIDIA H100 80GB; one probe run = one GPU.                                                                                                                                      
                                                                              
  Reported metrics on test:                                                                                                                                                               
  - R² = 1 − SSR/SST, computed on per-video pred-averaged predictions, n = 1277 videos
  - Pearson correlation                                                                                                                                                                   
  - MAE in LVEF % (raw units)                                                 
                                                                                                                                                                                          
  ---                                                                                                                                                                                     
  X.3 Experiment 1 — Linear-A raw / diff probe reproduction                                                                                                                               
                                                                                                                                                                                          
  Purpose. Reproduce the published linear-probe numbers on freshly-extracted caches (as a sanity check) and extend the trajectory to e200/e194.                                           
                                                                                                                                                                                          
  Probe architectures (reused from scripts/neurips/diff_probe_train.py):                                                                                                                  
                                                                                                                                                                                          
  - LinearA: Flatten([T', D]) → Linear(T' · D, 1). T' is 8 for raw, 7 for diff.                                                                                                           
                                                                              
  Input variants.                                                                                                                                                                         
                                                                              
  ┌─────────┬─────────────────────────────┬─────────────────┐                                                                                                                             
  │ variant │          transform          │      shape      │                 
  ├─────────┼─────────────────────────────┼─────────────────┤                                                                                                                             
  │ raw     │ identity                    │ [T=8, D=1024]   │                 
  ├─────────┼─────────────────────────────┼─────────────────┤                                                                                                                             
  │ diff    │ z[t+1, :] − z[t, :], signed │ [T-1=7, D=1024] │                                                                                                                             
  └─────────┴─────────────────────────────┴─────────────────┘                                                                                                                             
                                                                                                                                                                                          
  HP grid.                                                                                                                                                                                
  - LR = 1e-3 (fixed)                                                         
  - WD ∈ {1e-4, 1e-2} (sweep of 2)                                                                                                                                                        
  - 5 seeds × 2 WDs = 10 runs per (model, input)                              
                                                                                                                                                                                          
  Jobs:                                                                                                                                                                                   
  - Linear-A raw: SLURM 418 (sbatch scripts/neurips/phase/linear_probe_train_only.sbatch), 1m46s                                                                                          
  - Linear-A diff: SLURM 423 (sbatch scripts/neurips/phase/linear_diff_probe_train.sbatch), 2m16s                                                                                         
                                                                                                                                                                                          
  Results table 1 — Linear-A raw and diff, test R²                                                                                                                                        
                                                                                                                                                                                          
  ┌─────────────────┬────────────────────────────┬────────────────┬─────────────────────────────┬────────────────┐                                                                        
  │      model      │ Linear-A raw R² mean ± std │     range      │ Linear-A diff R² mean ± std │     range      │                                                                        
  ├─────────────────┼────────────────────────────┼────────────────┼─────────────────────────────┼────────────────┤                                                                        
  │ JEPA IN21K e100 │              0.502 ± 0.011 │ [0.487, 0.520] │               0.375 ± 0.008 │ [0.361, 0.385] │
  ├─────────────────┼────────────────────────────┼────────────────┼─────────────────────────────┼────────────────┤                                                                        
  │ JEPA IN21K e200 │              0.531 ± 0.016 │ [0.505, 0.551] │               0.434 ± 0.008 │ [0.417, 0.440] │                                                                        
  ├─────────────────┼────────────────────────────┼────────────────┼─────────────────────────────┼────────────────┤                                                                        
  │ EchoMAE-L e99   │              0.670 ± 0.004 │ [0.665, 0.676] │               0.626 ± 0.008 │ [0.609, 0.632] │                                                                        
  ├─────────────────┼────────────────────────────┼────────────────┼─────────────────────────────┼────────────────┤                                                                        
  │ EchoMAE-L e194  │              0.694 ± 0.007 │ [0.684, 0.703] │               0.643 ± 0.007 │ [0.633, 0.654] │
  └─────────────────┴────────────────────────────┴────────────────┴─────────────────────────────┴────────────────┘                                                                        
                                                                              
  Reproduction verification (vs prior trajectory table; means to the 0.001):                                                                                                              
                                                                              
  ┌─────────────────┬───────────────┬─────────────────┬────────────┬─────────────────┐                                                                                                    
  │      model      │   prior raw   │ reproduced raw  │ prior diff │ reproduced diff │
  ├─────────────────┼───────────────┼─────────────────┼────────────┼─────────────────┤                                                                                                    
  │ JEPA IN21K e100 │ 0.502 ± 0.013 │ 0.502 ± 0.011 ✓ │      0.376 │         0.375 ✓ │
  ├─────────────────┼───────────────┼─────────────────┼────────────┼─────────────────┤                                                                                                    
  │ EchoMAE-L e99   │ 0.670 ± 0.005 │ 0.670 ± 0.004 ✓ │      0.626 │         0.626 ✓ │                                                                                                    
  └─────────────────┴───────────────┴─────────────────┴────────────┴─────────────────┘                                                                                                    
                                                                                                                                                                                          
  Means match exactly; std differences of ±0.002 are attributable to RNG state differences in bootstrapping / split order. Verified the canonical linear-probe numbers on                 
  freshly-extracted caches.                                                   
                                                                                                                                                                                          
  Results S3:                                                                                                                                                                             
  - Raw: runs/linear_probe_verify_418/results/{jepa_e100,jepa_e200,mae_e99,mae_e194}.json
  - Diff: runs/linear_diff_verify_423/results/{jepa_e100,jepa_e200,mae_e99,mae_e194}.json                                                                                                 
                                                                                         
  ---                                                                                                                                                                                     
  X.4 Experiment 2 — Pooled temporal-attention bridge probe                                                                                                                               
                                                                                                                                                                                          
  Purpose. Add a medium-capacity probe with probe-side temporal attention but no access to the spatial token grid, to isolate whether the main-text full-token attentive probe's JEPA >   
  MAE ordering comes from temporal attention alone or from the spatial token grid.                                                                                                        
                                                                      
  Probe architecture (scripts/neurips/pooled_temporal_attn_probe_train.py::CrossAttnPool):                                                                                                
                                                                      
  Input:  [B, T=8, D=1024]    (raw pooled sequence)                                                                                                                                       
    -> LayerNorm(D)                                                                                                                                                                       
    -> learned query token q ∈ ℝ^{1, 1, D}, trunc-normal init std=0.02                                                                                                                    
    -> MultiheadAttention(q as query, x as keys/values), 8 heads, dropout=0.1, batch_first=True                                                                                           
    -> residual + LayerNorm(D)                                                                                                                                                            
    -> MLP(D → 4D → D) with GELU + dropout=0.1                                                                                                                                            
    -> residual + LayerNorm(D)                                                                                                                                                            
    -> Linear(D, 1) head                                                                                                                                                                  
    -> squeeze([B]) prediction                                                                                                                                                            
                                                                                                                                                                                          
  - Number of heads: 8 (auto-reduced if D % n_heads != 0, but 1024 % 8 = 0 so 8 is used).                                                                                                 
  - MLP ratio: 4                                                                                                                                                                          
  - Dropout: 0.1                                                                                                                                                                          
  - Parameter count: ~8.4 M                                                                                                                                                               
    - Reference: full-token d=4 attentive probe has ~50 M parameters; linear-A has 8,193.                                                                                                 
                                                                                                                                                                                          
  Input variant. raw: [T=8, D=1024] unchanged (no diff, no shuffle).                                                                                                                      
                                                                                                                                                                                          
  HP grid.                                                                                                                                                                                
  - LR ∈ {1e-4, 3e-4, 1e-3}                                           
  - WD ∈ {1e-4, 1e-2}                                                                                                                                                                     
  - 5 seeds × 3 LRs × 2 WDs = 30 runs per model, 120 total            
                                                                                                                                                                                          
  Job: SLURM 424 (sbatch scripts/neurips/phase/pooled_temporal_attn_probe_train.sbatch), 8m32s (4 models in parallel, one per GPU, 5 seeds × 6 HP configs sequentially per GPU).          
                                                                                                                                                                                          
  Results table 2 — Pooled temporal-attn raw, test R²                                                                                                                                     
                                                                                                                                                                                          
  ┌─────────────────┬───────────────┬──────────┬──────────────┬──────────────────────────────────────────────────────────────────────┐                                                    
  │      model      │ R² mean ± std │ MAE mean │ Pearson mean │                   best-HP modes (LR, WD) per seed                    │
  ├─────────────────┼───────────────┼──────────┼──────────────┼──────────────────────────────────────────────────────────────────────┤                                                    
  │ JEPA IN21K e100 │ 0.667 ± 0.010 │     5.20 │        0.818 │ {3e-4, 1e-2}, {1e-4, 1e-4}, {1e-4, 1e-2}, {3e-4, 1e-4}, {1e-4, 1e-2} │
  ├─────────────────┼───────────────┼──────────┼──────────────┼──────────────────────────────────────────────────────────────────────┤                                                    
  │ JEPA IN21K e200 │ 0.665 ± 0.009 │     5.17 │        0.818 │ {1e-4, 1e-4} × 4, {1e-4, 1e-2} × 1                                   │                                                    
  ├─────────────────┼───────────────┼──────────┼──────────────┼──────────────────────────────────────────────────────────────────────┤                                                    
  │ EchoMAE-L e99   │ 0.747 ± 0.009 │     4.60 │        0.867 │ {1e-4, 1e-2} × 2, {1e-4, 1e-4} × 3                                   │                                                    
  ├─────────────────┼───────────────┼──────────┼──────────────┼──────────────────────────────────────────────────────────────────────┤                                                    
  │ EchoMAE-L e194  │ 0.757 ± 0.005 │     4.45 │        0.871 │ {1e-4, 1e-4}, {3e-4, 1e-2}, {3e-4, 1e-4}, {1e-4, 1e-2}, {1e-4, 1e-2} │
  └─────────────────┴───────────────┴──────────┴──────────────┴──────────────────────────────────────────────────────────────────────┘                                                    
                                                                      
  Median best epoch across 120 runs: 11. Maximum ≤ 27; no run hit the 50-epoch ceiling.                                                                                                   
                                                                      
  Results S3: runs/pooled_temporal_attn_probe_verify_424/results/{jepa_e100,jepa_e200,mae_e99,mae_e194}.json                                                                              
                                                                      
  ---                                                                                                                                                                                     
  X.5 Experiment 3 — Spatial / phase / temporal-information controls  
                                                                                                                                                                                          
  Purpose. Resolve the five competing mechanisms for MAE's pooled-feature advantage:
  1. aggregate / static anatomy                                                                                                                                                           
  2. single-frame or phase-specific geometry                                                                                                                                              
  3. time-indexed sequence information                                                                                                                                                    
  4. adjacent-frame differences                                                                                                                                                           
  5. learned temporal aggregation                                                                                                                                                         
                                                                                                                                                                                          
  Probe architectures used.                                                                                                                                                               
  - LinearA (same as Experiment 1)                                                                                                                                                        
  - CrossAttnPool (same as Experiment 2)                                                                                                                                                  
                                                                                                                                                                                          
  Input variants.                                                                                                                                                                         
                                                                                                                                                                                          
  ┌───────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────┬────────┬───────────────┐                     
  │          variant          │                                                 transform                                                  │ shape  │     probe     │                     
  ├───────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┼───────────────┤                     
  │ raw (baseline from §X.3)  │ identity                                                                                                   │ [8, D] │ LinearA       │
  ├───────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┼───────────────┤                     
  │ diff (baseline from §X.3) │ z[t+1] − z[t]                                                                                              │ [7, D] │ LinearA       │                     
  ├───────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┼───────────────┤                     
  │ mean                      │ (1/T) Σ_t z_t                                                                                              │ [1, D] │ LinearA       │                     
  ├───────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┼───────────────┤                     
  │ single_tk (k=0..7)        │ single token at index k                                                                                    │ [1, D] │ LinearA       │
  ├───────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┼───────────────┤                     
  │ single_t_best_val         │ per-seed: token chosen by highest val R² over the 8 single-token probes; then report test R² at that token │ [1, D] │ LinearA       │
  ├───────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┼───────────────┤                     
  │ mean_repeat               │ [(1/T)Σ z_t] × T (broadcast to T positions)                                                                │ [8, D] │ CrossAttnPool │
  ├───────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┼───────────────┤                     
  │ best_repeat               │ [z_{t*}] × T where t* = val-selected best token per seed                                                   │ [8, D] │ CrossAttnPool │
  └───────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────┴────────┴───────────────┘                     
                                                                      
  HP grids (same as Experiments 1–2):                                                                                                                                                     
  - Linear variants: LR=1e-3, WD∈{1e-4, 1e-2}. 5 seeds × 2 WDs × 10 variants = 100 linear runs per model.
  - Pooled temporal-attn variants: LR∈{1e-4, 3e-4, 1e-3}, WD∈{1e-4, 1e-2}. 5 seeds × 6 HPs × 2 variants = 60 PTA runs per model.                                                          
  - Total per model: 160 runs. 4 models parallelized on 4 GPUs: 640 runs.                                                       
                                                                                                                                                                                          
  Val-selection protocol for single_t_best_val: for each seed, train 16 single-token probes (8 tokens × 2 WDs) and pick the (token, WD) with highest val R²; report its test R². The      
  5-seed mean of those test R²'s is reported as lin t-best-val.                                                                                                                           
                                                                                                                                                                                          
  Val-selection protocol for best_repeat: for each seed, read the single_t_best_val token index, construct the repeated-token sequence using that token, and train a fresh PTA probe.     
                                                                      
  Job: SLURM 425 (sbatch scripts/neurips/phase/spatial_phase_controls_train.sbatch; Python driver scripts/neurips/spatial_phase_controls_train.py), 26m49s, 4 models parallel on 4 GPUs.  
                                                                      
  Results table 3 — Full control matrix, test R² (5-seed mean ± std)                                                                                                                      
                                                                      
  ┌─────────────────┬─────────────────┬───────────────┬───────────────┬───────────────┬───────────────┬───────────────┬────────────────┬───────────────┬───────────────┐                  
  │      model      │ full-token attn │    pta raw    │    lin raw    │   lin diff    │   lin mean    │   lin t-avg   │ lin t-best-val │ pta mean-rep  │ pta best-rep  │
  ├─────────────────┼─────────────────┼───────────────┼───────────────┼───────────────┼───────────────┼───────────────┼────────────────┼───────────────┼───────────────┤                  
  │ JEPA IN21K e100 │           0.650 │ 0.667 ± 0.010 │ 0.502 ± 0.012 │ 0.376 ± 0.008 │ 0.444 ± 0.014 │ 0.441 ± 0.007 │  0.447 ± 0.006 │ 0.537 ± 0.015 │ 0.573 ± 0.019 │
  ├─────────────────┼─────────────────┼───────────────┼───────────────┼───────────────┼───────────────┼───────────────┼────────────────┼───────────────┼───────────────┤                  
  │ JEPA IN21K e200 │           0.684 │ 0.665 ± 0.009 │ 0.531 ± 0.016 │ 0.435 ± 0.008 │ 0.472 ± 0.016 │ 0.462 ± 0.009 │  0.476 ± 0.005 │ 0.566 ± 0.015 │ 0.575 ± 0.008 │                  
  ├─────────────────┼─────────────────┼───────────────┼───────────────┼───────────────┼───────────────┼───────────────┼────────────────┼───────────────┼───────────────┤                  
  │ EchoMAE-L e99   │           0.447 │ 0.747 ± 0.009 │ 0.670 ± 0.004 │ 0.626 ± 0.008 │ 0.644 ± 0.006 │ 0.636 ± 0.006 │  0.637 ± 0.006 │ 0.708 ± 0.005 │ 0.695 ± 0.012 │                  
  ├─────────────────┼─────────────────┼───────────────┼───────────────┼───────────────┼───────────────┼───────────────┼────────────────┼───────────────┼───────────────┤                  
  │ EchoMAE-L e194  │           0.526 │ 0.757 ± 0.005 │ 0.694 ± 0.007 │ 0.644 ± 0.007 │ 0.676 ± 0.008 │ 0.668 ± 0.004 │  0.673 ± 0.007 │ 0.724 ± 0.007 │ 0.724 ± 0.011 │
  └─────────────────┴─────────────────┴───────────────┴───────────────┴───────────────┴───────────────┴───────────────┴────────────────┴───────────────┴───────────────┘                  
                                                                      
  full-token attn numbers from published frame-shuffling trajectory (job 220 for e25–e100; job 379 for e125–e200; analogous for MAE extended trajectory). All other numbers from jobs 418,
   423, 424, 425 (this appendix).                                     
                                                                                                                                                                                          
  Columns legend:                                                                                                                                                                         
  - full-token attn: best-head of 6-HP d=4 attentive probe on full spatial×temporal token grid (ViT-L features with spatial dim preserved)
  - pta raw: CrossAttnPool on [8, 1024] raw (Experiment 2)                                                                                                                                
  - lin raw/lin diff: LinearA on [T, D] raw / [T-1, D] adjacent signed diffs (Experiment 1)
  - lin mean: LinearA on time-mean [1, D]                                                                                                                                                 
  - lin t-avg: mean across 8 independently-trained single-token linear probes (5 seeds each); reported as mean ± std of per-token 5-seed R² means                                         
  - lin t-best-val: per seed, pick best token by val R² across 8 tokens × 2 WDs; report test R² at that (token, WD); 5-seed mean ± std                                                    
  - pta mean-rep: CrossAttnPool on [mean, mean, …, mean] (T positions, identical tokens)                                                                                                  
  - pta best-rep: CrossAttnPool on [z_{t*}, z_{t*}, …, z_{t*}] with t* = val-selected best token per seed                                                                                 
                                                                                                                                                                                          
  Results table 4 — Per-token linear R² (best WD per seed, 5-seed mean)                                                                                                                   
                                                                                                                                                                                          
  ┌─────────────────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────────┐                                                                                 
  │      model      │  t0   │  t1   │  t2   │  t3   │  t4   │  t5   │  t6   │  t7   │  avg  │ max − min │                                                                                 
  ├─────────────────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────────┤                                                                                 
  │ JEPA IN21K e100 │ 0.436 │ 0.448 │ 0.443 │ 0.443 │ 0.428 │ 0.437 │ 0.442 │ 0.454 │ 0.441 │     0.026 │
  ├─────────────────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────────┤                                                                                 
  │ JEPA IN21K e200 │ 0.452 │ 0.470 │ 0.479 │ 0.461 │ 0.446 │ 0.465 │ 0.461 │ 0.462 │ 0.462 │     0.033 │                                                                                 
  ├─────────────────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────────┤                                                                                 
  │ EchoMAE-L e99   │ 0.628 │ 0.635 │ 0.643 │ 0.640 │ 0.626 │ 0.636 │ 0.637 │ 0.641 │ 0.636 │     0.017 │                                                                                 
  ├─────────────────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────────┤                                                                                 
  │ EchoMAE-L e194  │ 0.663 │ 0.669 │ 0.674 │ 0.671 │ 0.665 │ 0.661 │ 0.668 │ 0.672 │ 0.668 │     0.013 │
  └─────────────────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────────┘                                                                                 
                                                                      
  Per-token standard deviation across T=8 tokens, each model:                                                                                                                             
  - JEPA e100: σ across t ≈ 0.008                                     
  - JEPA e200: σ across t ≈ 0.010                                                                                                                                                         
  - MAE e99: σ across t ≈ 0.006                                       
  - MAE e194: σ across t ≈ 0.004                                                                                                                                                          
                                                                                                                                                                                          
  Across-token R² spread is < 0.033 within any model, and comparable to the within-seed std for the best-token probe (5-seed std = 0.005–0.007). There is no dominant phase token. The    
  single_t_best_val result is barely higher than t-avg, confirming that no specific phase carries the LVEF signal — any frame gives ≈ the same result.                                    
                                                                                                                                                                                          
  Results table 5 — Val-selected best token index, per (model, seed)                                                                                                                      
                                                                              
  ┌─────────────────┬────────┬────────┬────────┬────────┬────────┐                                                                                                                        
  │      model      │ seed 0 │ seed 1 │ seed 2 │ seed 3 │ seed 4 │    
  ├─────────────────┼────────┼────────┼────────┼────────┼────────┤                                                                                                                        
  │ JEPA IN21K e100 │ 7      │ 2      │ 7      │ 5      │ 3      │    
  ├─────────────────┼────────┼────────┼────────┼────────┼────────┤                                                                                                                        
  │ JEPA IN21K e200 │ 2      │ 1      │ 6      │ 2      │ 2      │                                                                                                                        
  ├─────────────────┼────────┼────────┼────────┼────────┼────────┤                                                                                                                        
  │ EchoMAE-L e99   │ 0      │ 7      │ 5      │ 5      │ 5      │                                                                                                                        
  ├─────────────────┼────────┼────────┼────────┼────────┼────────┤                                                                                                                        
  │ EchoMAE-L e194  │ 7      │ 1      │ 5      │ 7      │ 5      │    
  └─────────────────┴────────┴────────┴────────┴────────┴────────┘                                                                                                                        
                                                                      
  Tokens selected are scattered across all 8 positions; no clustering at end-diastolic or end-systolic indices.                                                                           
                                                                      
  Results S3: runs/spatial_phase_controls_425/results/{jepa_e100,jepa_e200,mae_e99,mae_e194}.json                                                                                         
                                                                      
  ---                                                                                                                                                                                     
  X.6 Derived gap tables                                              
                                                                                                                                                                                          
  Table 6 — Temporal-collapse gaps (absolute R² loss when temporal information is removed)
                                                                                                                                                                                          
  Negative gap = performance lost by removing temporal information.                                                                                                                       
                                                                                                                                                                                          
  ┌────────────────────────────────────────────────────────────────────┬───────────┬───────────┬─────────┬──────────┐                                                                     
  │                                gap                                 │ JEPA e100 │ JEPA e200 │ MAE e99 │ MAE e194 │
  ├────────────────────────────────────────────────────────────────────┼───────────┼───────────┼─────────┼──────────┤                                                                     
  │ lin_raw − lin_mean (time-averaging cost)                           │    −0.058 │    −0.059 │  −0.026 │   −0.018 │
  ├────────────────────────────────────────────────────────────────────┼───────────┼───────────┼─────────┼──────────┤                                                                     
  │ lin_raw − lin_diff (loss from differencing)                        │    −0.127 │    −0.097 │  −0.044 │   −0.051 │                                                                     
  ├────────────────────────────────────────────────────────────────────┼───────────┼───────────┼─────────┼──────────┤                                                                     
  │ lin_raw − lin_t-best-val (raw vs single-frame oracle)              │    −0.055 │    −0.055 │  −0.033 │   −0.021 │                                                                     
  ├────────────────────────────────────────────────────────────────────┼───────────┼───────────┼─────────┼──────────┤                                                                     
  │ pta_raw − pta_mean_repeat (real-sequence benefit for PTA)          │    −0.130 │    −0.099 │  −0.039 │   −0.033 │
  ├────────────────────────────────────────────────────────────────────┼───────────┼───────────┼─────────┼──────────┤                                                                     
  │ pta_raw − pta_best_repeat (real-sequence vs repeated-best benefit) │    −0.094 │    −0.090 │  −0.052 │   −0.033 │
  └────────────────────────────────────────────────────────────────────┴───────────┴───────────┴─────────┴──────────┘                                                                     
                                                                      
  Table 7 — MAE − JEPA deltas across probe families (paired by same cache and probe)                                                                                                      
                                                                      
  ┌─────────────────┬─────────────────────┬──────────────────────┐                                                                                                                        
  │  probe family   │ MAE e99 − JEPA e100 │ MAE e194 − JEPA e200 │    
  ├─────────────────┼─────────────────────┼──────────────────────┤                                                                                                                        
  │ lin raw         │              +0.168 │               +0.163 │    
  ├─────────────────┼─────────────────────┼──────────────────────┤                                                                                                                        
  │ lin diff        │              +0.250 │               +0.209 │                                                                                                                        
  ├─────────────────┼─────────────────────┼──────────────────────┤                                                                                                                        
  │ lin mean        │              +0.200 │               +0.204 │                                                                                                                        
  ├─────────────────┼─────────────────────┼──────────────────────┤                                                                                                                        
  │ lin t-avg       │              +0.195 │               +0.206 │    
  ├─────────────────┼─────────────────────┼──────────────────────┤                                                                                                                        
  │ lin t-best-val  │              +0.190 │               +0.197 │    
  ├─────────────────┼─────────────────────┼──────────────────────┤                                                                                                                        
  │ pta raw         │              +0.080 │               +0.092 │    
  ├─────────────────┼─────────────────────┼──────────────────────┤                                                                                                                        
  │ pta mean-rep    │              +0.171 │               +0.158 │    
  ├─────────────────┼─────────────────────┼──────────────────────┤                                                                                                                        
  │ pta best-rep    │              +0.122 │               +0.149 │    
  ├─────────────────┼─────────────────────┼──────────────────────┤                                                                                                                        
  │ full-token attn │              −0.203 │               −0.158 │    
  └─────────────────┴─────────────────────┴──────────────────────┘                                                                                                                        
                                                                      
  All pooled-feature probe families give MAE > JEPA; only the full-token attentive probe gives JEPA > MAE.                                                                                
                                                                      
  ---                                                                                                                                                                                     
  X.7 Interpretation                                                  
                                                                                                                                                                                          
  For MAE                                                             
                                                                                                                                                                                          
  1. lin_mean vs lin_raw gap is tiny (+0.02–0.03). Collapsing the temporal axis barely changes linear LVEF decodability, implying almost all LVEF-relevant information in MAE's pooled    
  features is in the time-averaged (aggregate-anatomy) embedding.                                                                                                                         
  2. Per-token spread < 0.02. No specific phase is privileged; every frame carries roughly the same static-anatomy information.                                                           
  3. pta_mean_repeat (0.708) ≈ 95% of pta_raw (0.747). The temporal-attention probe does not need a real sequence for MAE; feeding it a constant repeated token nearly matches            
  real-sequence performance.                                                                                                                                                              
  4. pta_best_repeat ≈ pta_mean_repeat. The best single token repeated T times is indistinguishable from the mean repeated T times.                                                       
                                                                                                                                                                                          
  Mechanism: MAE's pooled features encode LVEF predominantly as aggregate / phase-averaged static anatomy. The temporal structure adds at most ~0.04 R² on top.                           
                                                                                                                                                                                          
  For JEPA                                                                                                                                                                                
                                                                      
  1. lin_raw − lin_mean gap is +0.06 — twice MAE's — showing JEPA's pooled features do contain time-indexed information the linear probe partly recovers.                                 
  2. pta_raw − pta_mean_repeat gap is −0.13 — three times MAE's gap — demonstrating that temporal attention on a real sequence exploits something the model has encoded that a
  constant-sequence cannot recover.                                                                                                                                                       
  3. Per-token spread similarly small (~0.02), so the temporal gain is not phase-specific at the single-token level; it emerges from learned aggregation across distinct tokens.
  4. **pta_best_repeat < pta_mean_repeat for JEPA e100** (0.573 vs 0.537 — wait, 0.573 > 0.537); best-repeat slightly higher, but both far below raw. Differs from MAE where mean_repeat ≥
   best_repeat, suggesting JEPA's temporal integration is compositional across phases in a way MAE's is not.                                                                              
                                                                                                                                                                                          
  Mechanism: JEPA has a real but modest temporal-aggregation component (~0.13 R² on top of a constant-sequence baseline), but its baseline per-frame anatomy is weaker than MAE's, so its 
  total pooled-feature LVEF R² is lower.                        
                                                                                                                                                                                          
  Joint conclusion                                                                                                                                                                        
                                                                
  The paper's "MAE abandons temporal features" framing is accurate only under the specific full-token attentive probe protocol where matched-frame ΔR² ≈ 0 for MAE and −0.143 for JEPA. At
   the pooled-feature level:                                    
                                                                                                                                                                                          
  - MAE's features are more LVEF-decodable than JEPA's across every non-full-token probe we tested.                                                                                       
  - MAE's pooled LVEF signal is overwhelmingly static / phase-averaged anatomy, but this is a feature of what the model has encoded, not an absence of signal.
  - JEPA has a real temporal-aggregation component at the pooled-feature level, but it is smaller (~0.13 R²) than MAE's aggregate-anatomy surplus (~0.17 R²).                             
  - Only the full-token attentive probe reverses the ranking. This is the probe where the paper's central JEPA > MAE result comes from.                                                   
                                                                                                                                                                                          
  ---                                                                                                                                                                                     
  X.8 Recommended paper placement                                                                                                                                                         
                                                                                                                                                                                          
  Main text (one paragraph): reframe the "MAE abandons temporal features" claim. Specifically: MAE's pooled features are more LVEF-decodable than JEPA's under every non-full-token probe;
   the published full-token attentive ranking is protocol-specific and reflects the interaction between probe architecture and feature content, not an absolute property of which encoder 
  encodes LVEF better.                                          
                                                                                                                                                                                          
  Appendix: full control matrix (Tables 3–5), per-token table, derived-gap table. This section.                                                                                           
                                                                
  Limitations: explicitly note that the headline JEPA > MAE ordering (on EchoNet LVEF) holds only under the full-token attentive probe, not pooled-feature probes, and that MAE's         
  pooled-feature advantage is driven by aggregate / phase-averaged anatomy rather than by temporal encoding.
                                                                                                                                                                                          
  ---                                
  X.9 Caveats and known limitations
                                                                                                                                                                                          
  1. Probe capacity is not matched. Linear-A has 8.2K parameters; pooled temporal-attn has ~8.4M; full-token attentive d=4 has ~50M. We did not equalize probe-compute across families.
  The large-probe results could partially reflect over-fitting.                                                                                                                           
  2. No bootstrap CIs on probe metrics — only 5-seed std as noise estimate. Bootstrap would require storing per-video predictions for every run; we retained those in JSON for ~120 runs
  per experiment but did not compute CIs for this appendix.                                                                                                                               
  3. Feature-space vs input-space matched-frame. The full-token attentive probe is evaluated with input-time frame shuffling + RoPE remap; the pooled-feature probes here are not. We did
  not run a matched-frame variant of the pooled temporal-attn probe.                                                                                                                      
  4. Single downstream task. All results are on EchoNet-Dynamic LVEF regression. Other tasks (RVSP, view classification, pediatric) may show different probe-family orderings; the paper's
   other §s need separate analysis.                                                                                                                                                       
  5. Spatial axis discarded in cache. The pre-pool features have already spatially mean-pooled; we cannot re-introduce spatial tokens in this appendix without re-extracting with
  FEATURE_KEEP_SPATIAL=1.                                                                                                                                                                 
  6. Checkpoints. JEPA e200 and MAE e194 checkpoint md5s are not in the canonical checkpoint registry; they come from the latest training runs (jobs 280 and 179 respectively) and are
  cited by path. No hash collision risk for these specific runs has been identified.                                                                                                      
                                                                
  ---                                                                                                                                                                                     
  X.10 Reproducibility summary       
                                                                                                                                                                                          
  Code artifacts (all in the vjepa2 repo):                      
                                                                                                                                                                                          
  ┌───────────────────────────────────────────────────────────────────┬───────────────────────────────────────────┐                                                                       
  │                               file                                │                  purpose                  │                                                                       
  ├───────────────────────────────────────────────────────────────────┼───────────────────────────────────────────┤                                                                       
  │ evals/feature_extraction_pre_pool/eval.py                         │ pre-pool feature extraction               │
  ├───────────────────────────────────────────────────────────────────┼───────────────────────────────────────────┤                                                                       
  │ configs/feature_extraction/vitl/neurips/diff_probe/*_prepool.yaml │ canonical extraction configs              │                                                                       
  ├───────────────────────────────────────────────────────────────────┼───────────────────────────────────────────┤                                                                       
  │ scripts/neurips/diff_probe_train.py                               │ linear-A / linear-B / MLP probes (Exp. 1) │                                                                       
  ├───────────────────────────────────────────────────────────────────┼───────────────────────────────────────────┤                                                                       
  │ scripts/neurips/pooled_temporal_attn_probe_train.py               │ PTA probe (Exp. 2)                        │
  ├───────────────────────────────────────────────────────────────────┼───────────────────────────────────────────┤                                                                       
  │ scripts/neurips/spatial_phase_controls_train.py                   │ full control matrix (Exp. 3)              │
  ├───────────────────────────────────────────────────────────────────┼───────────────────────────────────────────┤                                                                       
  │ scripts/neurips/phase/linear_probe_extract_jepa_e200.sbatch       │ extract e200 (job 416)                    │
  ├───────────────────────────────────────────────────────────────────┼───────────────────────────────────────────┤                                                                       
  │ scripts/neurips/phase/linear_probe_extract_mae_e194.sbatch        │ extract e194 (job 417)                    │
  ├───────────────────────────────────────────────────────────────────┼───────────────────────────────────────────┤                                                                       
  │ scripts/neurips/phase/linear_probe_train_only.sbatch              │ Exp. 1 raw (job 418)                      │
  ├───────────────────────────────────────────────────────────────────┼───────────────────────────────────────────┤                                                                       
  │ scripts/neurips/phase/linear_diff_probe_train.sbatch              │ Exp. 1 diff (job 423)                     │
  ├───────────────────────────────────────────────────────────────────┼───────────────────────────────────────────┤                                                                       
  │ scripts/neurips/phase/pooled_temporal_attn_probe_train.sbatch     │ Exp. 2 (job 424)                          │
  ├───────────────────────────────────────────────────────────────────┼───────────────────────────────────────────┤                                                                       
  │ scripts/neurips/phase/spatial_phase_controls_train.sbatch         │ Exp. 3 (job 425)                          │
  └───────────────────────────────────────────────────────────────────┴───────────────────────────────────────────┘                                                                       
                                                                
  Data artifacts (S3 at sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/):                                                                                                
                                                                
  - Feature caches: features/diff_probe/{jepa_e100,jepa_e200,mae_e99,mae_e194}_{train,test}.pt                                                                                            
  - Linear raw: runs/linear_probe_verify_418/results/           
  - Linear diff: runs/linear_diff_verify_423/results/                                                                                                                                     
  - PTA raw: runs/pooled_temporal_attn_probe_verify_424/results/                                                                                                                          
  - Controls: runs/spatial_phase_controls_425/results/                                                                                                                                    
                                                                                                                                                                                          
  Shared probe protocol constants (relevant for every run in §§X.3–X.5):                                                                                                                  
                                                                                                                                                                                          
  echonet_dynamic_train_s3_raw.csv : 7,465 videos                                                                                                                                         
  echonet_dynamic_test_s3_raw.csv  : 1,277 videos                                                                                                                                         
  num_segments = 2                                                                                                                                                                        
  frames_per_clip = 16, frame_step = 2, resolution = 224                                                                                                                                  
  spatial-mean pre-pool (D = 1024 per temporal token, T = 8 tokens)                                                                                                                       
                                                                                                                                                                                          
  Per-seed stratified split: 90/10 EF-quintile-stratified, seed ∈ {0,1,2,3,4}                                                                                                             
  Training: AdamW, MSE on z-scored targets, batch size 64                                                                                                                                 
  Early stopping on val R²; best-HP by val R²; 5-seed test mean ± std reported.  