#!/usr/bin/env python3
"""Frozen cross-view retrieval diagnostic.

Purpose
-------
For a given frozen encoder checkpoint, measure how well the student's
*source*-view embedding retrieves the correct same-study *target*-view
embedding among a pool of other-study same-target-view candidates.

This is the eval that distinguishes

    "encoder learned cross-view physiology"            (generalises)

from

    "encoder's same-study PLAX is close to same-study A4C because both
     are same-patient / similar image statistics / study-level bias"
    (does not generalise to unseen studies).

We evaluate on HELD-OUT studies (test split) only. No training batches,
no overlap with the pretraining anchor pool.

Pipeline
--------
1. Build a ``PhaseMatchedStudySampler`` restricted to the study-disjoint
   test split (classifier/phase/splits/dicoms_split.csv, split=test).
2. For each ``MatchRecord`` with a valid target_clip (``source_view``
   anchor + ``target_view`` target_clip):
     - decode both 16-frame clips with the shared transforms,
     - pass through the frozen encoder (+ factorized head for MV2SV
       slot modes),
     - mean-pool over tokens to get a single vector per clip.
3. Group encoded targets by ``target_view`` (or view-family). For each
   source, compute cosine(source, same-study-target) (positive) and
   cosine(source, other-study-same-target-view-targets) (negatives).
4. Top-1 / top-5 retrieval within each target_view bucket; pooled
   across all source→target pairs.

Feature modes
-------------
    encoder_pool          — legacy mean-pooled encoder output
    z_shared / z_phase    — factorized head slot projections
    z_view                — view-specific residual slot
    concat_shared_phase   — [z_shared; z_phase]
    concat_all            — [z_shared; z_phase; z_view]

Mode ``encoder_pool`` works for ANY V-JEPA checkpoint (Base, SV, V3, V4,
control, no-hardneg ablation, MV2SV pilot/ctrl). Slot modes require an
MV2SV checkpoint with ``factorized_head_ema`` or ``factorized_head``
state saved.

Usage
-----
    python scripts/neurips/phase/run_cross_view_retrieval_diag.py \
        --checkpoint /opt/.../ckpt.pt \
        --feature-mode encoder_pool \
        --source-views A4C \
        --target-views A2C,A5C,A3C,PLAX,PSAX-MV \
        --num-studies 200 \
        --batch-size 16 \
        --out /tmp/retrieval_diag/<tag>.json

Outputs a JSON with:
    top1, top5, pos_sim, neg_sim, gap, valid_neg_mean/min, fallback_frac
    per (source_view, target_view) bucket and overall.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "classifier" / "phase" / "sampler"))


# --------------------------------------------------------------------------- #
# view-family mapping — mirrors app/vjepa_multiview/train.py::VIEW_FAMILY_ID
# --------------------------------------------------------------------------- #
_APICAL = {"A4C", "A5C", "A3C", "A2C"}
_PLAX_FAM = {"PLAX", "PSAX-MV", "PSAX-AV", "PSAX-PM", "PSAX-AP", "PSAX", "PLAX-RV"}


def _family(view: str) -> str:
    if view in _APICAL:
        return "apical"
    if view in _PLAX_FAM:
        return "parasternal"
    return "other"


# --------------------------------------------------------------------------- #
# Checkpoint + model loading
# --------------------------------------------------------------------------- #


def load_encoder_and_head(
    checkpoint_path: Path,
    feature_mode: str,
    resolution: int = 224,
    frames_per_clip: int = 16,
) -> tuple[torch.nn.Module, torch.nn.Module | None, int]:
    """Load frozen encoder + optional factorized head. Returns
    (encoder, head, embed_dim). Both are moved to CUDA and set to eval."""
    import src.models.vision_transformer as vit

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    # Standard V-JEPA checkpoints have "target_encoder" (EMA-teacher) and
    # "encoder" (student). We probe the teacher side — it is what the
    # MV2SV objective aligns against and is the protocol the paper's
    # other probes use.
    enc_key = "target_encoder" if "target_encoder" in ckpt else "encoder"
    enc_sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in ckpt[enc_key].items()}

    # ViT-L/16 with the standard V-JEPA configuration. We pull these
    # directly from the checkpoint; any MV2SV/phase-relational ckpt in
    # this family shares them.
    encoder = vit.vit_large(
        img_size=resolution,
        num_frames=frames_per_clip,
        patch_size=16,
        tubelet_size=2,
        uniform_power=True,
        use_rope=True,
    )
    msg = encoder.load_state_dict(enc_sd, strict=False)
    print(f"[load] encoder key='{enc_key}' missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    encoder.eval().cuda()
    for p in encoder.parameters():
        p.requires_grad = False

    head: torch.nn.Module | None = None
    embed_dim = encoder.embed_dim
    if feature_mode != "encoder_pool":
        from app.vjepa_multiview.factorized_head import FactorizedProjectionHead

        fh_key = "factorized_head_ema" if "factorized_head_ema" in ckpt else "factorized_head"
        if fh_key not in ckpt:
            raise KeyError(
                f"checkpoint {checkpoint_path} has no 'factorized_head_ema' or 'factorized_head' key; "
                f"use --feature-mode encoder_pool for non-MV2SV ckpts"
            )
        # These dims are fixed for our MV2SV configs.
        head = FactorizedProjectionHead(
            embed_dim=encoder.embed_dim,
            hidden_dim=1024,
            shared_dim=256,
            phase_dim=256,
            view_dim=256,
        )
        fh_sd = {k.replace("module.", ""): v for k, v in ckpt[fh_key].items()}
        head.load_state_dict(fh_sd, strict=False)
        head.eval().cuda()
        for p in head.parameters():
            p.requires_grad = False
        embed_dim = {
            "z_shared": 256,
            "z_phase": 256,
            "z_view": 256,
            "concat_shared_phase": 512,
            "concat_all": 768,
        }[feature_mode]
        print(f"[load] factorized_head key='{fh_key}' feature_mode={feature_mode} embed_dim={embed_dim}")

    return encoder, head, embed_dim


def _slot_from_pooled(head: torch.nn.Module, pooled: torch.Tensor, mode: str) -> torch.Tensor:
    with torch.no_grad():
        slots = head(pooled)
    if mode == "z_shared":
        return slots["z_shared"]
    if mode == "z_phase":
        return slots["z_phase"]
    if mode == "z_view":
        return slots["z_view"]
    if mode == "concat_shared_phase":
        return torch.cat([slots["z_shared"], slots["z_phase"]], dim=-1)
    if mode == "concat_all":
        return torch.cat([slots["z_shared"], slots["z_phase"], slots["z_view"]], dim=-1)
    raise ValueError(f"bad mode={mode!r}")


# --------------------------------------------------------------------------- #
# Sampler construction (held-out test split only)
# --------------------------------------------------------------------------- #


def _load_test_study_ids(splits_csv: Path) -> set[str]:
    import pandas as pd
    splits = pd.read_csv(splits_csv)
    return set(splits.loc[splits["split"] == "test", "study_id"].astype(str).unique())


def build_sampler(
    phase_annotations_path: Path,
    mv2sv_cfg: dict,
    view_pair_policy: dict,
    view_labels_path: Path,
    pairs_per_study: int = 4,
    min_view_confidence: float = 0.60,
    seed: int = 0,
):
    """Build a PhaseMatchedStudySampler with the standard MV2SV config.
    Records are filtered to held-out studies after build_records() by
    checking study_id against the splits CSV — the sampler itself does
    not expose a study_ids restriction, so we post-filter."""
    import pandas as pd
    from phase_matched_sampler import PhaseMatchedStudySampler  # type: ignore

    # Reconstruct view_labels / view_confidences dicts (sampler expects
    # these as kwargs rather than a path; mirrors data_manager.init_data).
    vdf = pd.read_csv(view_labels_path)
    if "dicom_id" not in vdf.columns and "s3_uri" in vdf.columns:
        vdf["dicom_id"] = vdf.s3_uri.astype(str).str.extract(r"/([^/]+)\.(?:mp4|dcm)$", expand=False)
    vdf = vdf.dropna(subset=["dicom_id"]).copy()
    vdf["dicom_id"] = vdf["dicom_id"].astype(str)
    view_labels = dict(zip(vdf["dicom_id"], vdf["view"].astype(str)))
    view_confidences = dict(zip(vdf["dicom_id"], vdf["view_confidence"].astype(float)))

    sampler = PhaseMatchedStudySampler(
        parquet_path=str(phase_annotations_path),
        tiers=("high", "medium"),
        require_rr_consistent=True,
        rr_filter_mode="strict",
        sampling_mode="uniform_phase",
        phase_tolerance=0.15,
        frames_per_clip=16,
        frame_step=1,
        pairs_per_study=pairs_per_study,
        same_session_only=False,
        resample_attempts=4,
        seed=seed,
        num_replicas=1,
        rank=0,
        view_labels=view_labels,
        view_confidences=view_confidences,
        min_view_confidence=min_view_confidence,
        total_epochs=1,
        view_pair_policy=view_pair_policy,
        delta_phase_mode="controlled_buckets",
        delta_phase_buckets=[0.0, 0.125, 0.25, 0.5],
        delta_phase_bucket_probs=[0.40, 0.30, 0.20, 0.10],
        require_same_study_wrong_phase_negative=False,
        mv2sv_config=mv2sv_cfg,
    )
    return sampler


# --------------------------------------------------------------------------- #
# Clip loading + encoding
# --------------------------------------------------------------------------- #


def _load_clip(uri: str, frame_indices: list[int], resolution: int = 224) -> torch.Tensor:
    """Return [C, T, H, W] float tensor in [0, 255]."""
    import decord
    import boto3
    import io

    if uri.startswith("s3://"):
        # Decord can't read S3 directly; download to a BytesIO.
        s3 = boto3.client("s3")
        assert uri.startswith("s3://")
        _, rest = uri.split("s3://", 1)
        bucket, key = rest.split("/", 1)
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read()
        vr = decord.VideoReader(io.BytesIO(data), ctx=decord.cpu(0))
    else:
        vr = decord.VideoReader(uri, ctx=decord.cpu(0))
    # Clamp frame indices to available range
    n = len(vr)
    frame_indices = [min(i, n - 1) for i in frame_indices]
    frames = vr.get_batch(frame_indices).asnumpy()  # [T, H, W, C]
    frames = torch.from_numpy(frames).float()
    # Resize to [resolution, resolution] if needed
    if frames.shape[1] != resolution or frames.shape[2] != resolution:
        frames = F.interpolate(
            frames.permute(0, 3, 1, 2),  # [T, C, H, W]
            size=(resolution, resolution),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)  # back to [T, H, W, C]
    # Normalize to [0, 1] and apply ImageNet stats (V-JEPA default)
    frames = frames / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3)
    frames = (frames - mean) / std
    frames = frames.permute(3, 0, 1, 2).contiguous()  # [C, T, H, W]
    return frames


def encode_clip(
    clip: torch.Tensor,
    encoder: torch.nn.Module,
    head: torch.nn.Module | None,
    feature_mode: str,
) -> torch.Tensor:
    """clip: [C, T, H, W]. Returns [embed_dim] (pooled)."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        x = clip.unsqueeze(0).cuda()  # [1, C, T, H, W]
        tokens = encoder(x)  # [1, N_tok, D]
        pooled = tokens.mean(dim=1)  # [1, D]
        if feature_mode == "encoder_pool":
            feat = pooled
        else:
            feat = _slot_from_pooled(head, pooled.float(), feature_mode)
    return feat.squeeze(0).float().cpu()


# --------------------------------------------------------------------------- #
# Retrieval metrics
# --------------------------------------------------------------------------- #


def compute_retrieval_metrics(
    sources: list[dict],  # each has {study_id, source_view, target_view, source_feat, target_feat}
    mode: str = "same_study_same_view",
) -> dict:
    """For each source row, compute retrieval metrics.

    mode:
      - "same_study_same_view": positive = same-study target clip;
        negatives = same-target-view candidates from OTHER studies.
        Measures: does the encoder retrieve the correct same-study
        target-view latent among same-view distractors?
      - "same_study_any_view": positive = same-study target clip;
        negatives = ALL other-study targets regardless of view.
        Measures: does the encoder group same-study clips tightly
        even without a view constraint? Higher-top1 than the
        same-view mode suggests same-study identity confound; if
        the same-view mode is the one that improves with method
        training, the learned signal is actually view-specific.
    """
    # Group by target_view for negative sampling.
    by_tgt = defaultdict(list)
    for i, r in enumerate(sources):
        by_tgt[r["target_view"]].append(i)

    # Normalize all embeddings
    src_mat = F.normalize(torch.stack([r["source_feat"] for r in sources]), dim=-1)
    tgt_mat = F.normalize(torch.stack([r["target_feat"] for r in sources]), dim=-1)

    # Per-row retrieval
    per_row = []
    fallback_used = 0
    for i, r in enumerate(sources):
        if mode == "same_study_any_view":
            cands = [j for j in range(len(sources)) if j != i and sources[j]["study_id"] != r["study_id"]]
        else:
            # same_study_same_view (default)
            cands = [j for j in by_tgt[r["target_view"]] if sources[j]["study_id"] != r["study_id"]]
            if len(cands) < 2:
                # family fallback — only applied in same-view mode
                fam = _family(r["target_view"])
                cands = [
                    j for j, s in enumerate(sources)
                    if s["study_id"] != r["study_id"] and _family(s["target_view"]) == fam
                ]
                if len(cands) >= 2:
                    fallback_used += 1
        if len(cands) < 1:
            continue
        # Pool = [positive, negatives...]. Positive is this row's own target.
        pool_idx = [i] + cands
        pool = tgt_mat[pool_idx]  # [n+1, D]
        src = src_mat[i]  # [D]
        sims = pool @ src  # [n+1]
        pos_sim = sims[0].item()
        neg_sims = sims[1:]
        rank = (neg_sims > pos_sim).sum().item()  # 0 = top-1
        per_row.append({
            "study_id": r["study_id"],
            "source_view": r["source_view"],
            "target_view": r["target_view"],
            "pos_sim": pos_sim,
            "neg_sim_mean": neg_sims.mean().item(),
            "neg_sim_max": neg_sims.max().item(),
            "n_negatives": len(cands),
            "rank": rank,  # 0-indexed
            "used_fallback": len(cands) != sum(1 for j in by_tgt[r["target_view"]] if j != i and sources[j]["study_id"] != r["study_id"]),
        })

    # Aggregate
    n = len(per_row)
    if n == 0:
        return {"n_rows": 0, "note": "no retrievable rows"}
    top1 = sum(1 for r in per_row if r["rank"] == 0) / n
    top5 = sum(1 for r in per_row if r["rank"] < 5) / n
    pos = np.mean([r["pos_sim"] for r in per_row])
    neg = np.mean([r["neg_sim_mean"] for r in per_row])
    neg_counts = [r["n_negatives"] for r in per_row]
    overall = {
        "n_rows": n,
        "top1": float(top1),
        "top5": float(top5),
        "pos_sim_mean": float(pos),
        "neg_sim_mean": float(neg),
        "gap": float(pos - neg),
        "valid_neg_count_mean": float(np.mean(neg_counts)),
        "valid_neg_count_min": int(np.min(neg_counts)),
        "fallback_fraction": float(sum(1 for r in per_row if r["used_fallback"]) / n),
    }
    # Per source-target bucket
    by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in per_row:
        by_pair[(r["source_view"], r["target_view"])].append(r)
    per_pair = {}
    for (s, t), rs in by_pair.items():
        m = len(rs)
        per_pair[f"{s}->{t}"] = {
            "n": m,
            "top1": float(sum(1 for r in rs if r["rank"] == 0) / m),
            "top5": float(sum(1 for r in rs if r["rank"] < 5) / m),
            "pos_sim_mean": float(np.mean([r["pos_sim"] for r in rs])),
            "neg_sim_mean": float(np.mean([r["neg_sim_mean"] for r in rs])),
            "gap": float(np.mean([r["pos_sim"] - r["neg_sim_mean"] for r in rs])),
        }
    return {"overall": overall, "per_pair": per_pair}


# --------------------------------------------------------------------------- #
# Slot geometry — held-out cosine stats across relationship types
# --------------------------------------------------------------------------- #


def compute_slot_geometry(sources: list[dict]) -> dict:
    """Average pairwise cosines across 4 relationship types on target
    embeddings:
      (A) same-study same-view  (n/a here; each row has one target clip, so
          same-study same-view pairs don't exist in our sample — instead we
          report the source↔target same-study cosine as a proxy)
      (B) same-study cross-view: source↔target where source_view != target_view
          (what the method is trained to make close)
      (C) other-study same-view: target↔target across studies, same view
          (distractor pool in retrieval)
      (D) other-study cross-view: target↔target across studies, different
          views (should be lowest if the encoder factorizes view + study)

    For MV2SV v5 to be encoding useful cross-view physiology (rather than
    same-study identity), B must be materially larger than D, and A (if
    computable) should not dwarf B.
    """
    tgt_mat = F.normalize(torch.stack([r["target_feat"] for r in sources]), dim=-1)
    src_mat = F.normalize(torch.stack([r["source_feat"] for r in sources]), dim=-1)

    # B: source↔target for each same-study row (training objective).
    #    We already have per-row pos_sim when retrieval was run — rec-compute
    #    explicitly here so slot_geometry is self-contained.
    same_study_src_tgt = []
    for i in range(len(sources)):
        same_study_src_tgt.append((src_mat[i] * tgt_mat[i]).sum().item())

    # C: target↔target, other-study, same target_view.
    other_same_view = []
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            if sources[i]["study_id"] == sources[j]["study_id"]:
                continue
            if sources[i]["target_view"] != sources[j]["target_view"]:
                continue
            other_same_view.append((tgt_mat[i] * tgt_mat[j]).sum().item())

    # D: target↔target, other-study, different target_view.
    other_diff_view = []
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            if sources[i]["study_id"] == sources[j]["study_id"]:
                continue
            if sources[i]["target_view"] == sources[j]["target_view"]:
                continue
            other_diff_view.append((tgt_mat[i] * tgt_mat[j]).sum().item())

    def _s(xs):
        if not xs:
            return {"n": 0}
        a = np.array(xs, dtype=np.float64)
        return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
                "p05": float(np.percentile(a, 5)), "p50": float(np.percentile(a, 50)),
                "p95": float(np.percentile(a, 95))}

    return {
        "B_same_study_cross_view_src_tgt": _s(same_study_src_tgt),
        "C_other_study_same_view_tgt_tgt": _s(other_same_view),
        "D_other_study_diff_view_tgt_tgt": _s(other_diff_view),
        # headline scalars
        "B_minus_C_mean": float(np.mean(same_study_src_tgt) - np.mean(other_same_view)) if other_same_view else None,
        "B_minus_D_mean": float(np.mean(same_study_src_tgt) - np.mean(other_diff_view)) if other_diff_view else None,
        "C_minus_D_mean": float(np.mean(other_same_view) - np.mean(other_diff_view)) if (other_same_view and other_diff_view) else None,
        "note": (
            "B = same-study cross-view (training objective); "
            "C = other-study same-view; "
            "D = other-study different-view. "
            "Healthy pattern: B > C > D. Pure view clustering without study signal: C > B. "
            "Same-study identity confound: B >> C ≈ D."
        ),
    }


# --------------------------------------------------------------------------- #
# View classifier — catches the "pilot learned view classifier" failure mode
# --------------------------------------------------------------------------- #


def train_view_classifier(sources: list[dict]) -> dict:
    """Train a 1-layer linear probe to predict target_view from the
    encoded target embedding. 80/20 study-disjoint split within the
    test-split sample. Reports val accuracy, per-class recall, and
    confusion counts.

    Signal: if pilot 655 trivially nails view classification (acc >=
    0.95) while RVSP/MR stays null, the encoder is preferentially
    learning a view classifier rather than hallucinating target-view
    physiology. Base / V4 should be ~0.70-0.85 (view info is
    implicit); pilot > 0.95 with RVSP null is the red flag.
    """
    # Build (feat, label) from target embeddings only. Encode view string -> int.
    view_ids = sorted({r["target_view"] for r in sources})
    view_to_i = {v: i for i, v in enumerate(view_ids)}
    feats = torch.stack([r["target_feat"] for r in sources]).float()
    labels = torch.tensor([view_to_i[r["target_view"]] for r in sources], dtype=torch.long)
    studies = [r["study_id"] for r in sources]

    # Study-disjoint 80/20 split
    uniq = sorted(set(studies))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(uniq))
    n_train = int(0.8 * len(uniq))
    train_studies = set(uniq[i] for i in perm[:n_train])
    train_mask = torch.tensor([s in train_studies for s in studies])
    val_mask = ~train_mask
    if train_mask.sum() < 10 or val_mask.sum() < 10:
        return {"n_classes": len(view_ids), "n_train": int(train_mask.sum()),
                "n_val": int(val_mask.sum()), "note": "too few samples for split"}

    # Standardize features (encoder_pool is ~1024-D, slots are 256-D)
    mean = feats[train_mask].mean(dim=0, keepdim=True)
    std = feats[train_mask].std(dim=0, keepdim=True).clamp(min=1e-6)
    feats_std = (feats - mean) / std

    D = feats.shape[1]
    C = len(view_ids)
    # Small MLP-free linear probe with weight decay; CPU is fine at this size.
    head = torch.nn.Linear(D, C)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    X_tr, y_tr = feats_std[train_mask], labels[train_mask]
    X_va, y_va = feats_std[val_mask], labels[val_mask]
    best_val = 0.0
    for ep in range(200):
        head.train()
        perm2 = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), 256):
            idx = perm2[i : i + 256]
            opt.zero_grad()
            logits = head(X_tr[idx])
            loss = loss_fn(logits, y_tr[idx])
            loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            val_logits = head(X_va)
            val_pred = val_logits.argmax(dim=-1)
            val_acc = float((val_pred == y_va).float().mean())
        best_val = max(best_val, val_acc)

    # Final per-class stats
    head.eval()
    with torch.no_grad():
        val_logits = head(X_va)
        val_pred = val_logits.argmax(dim=-1)
    cm = torch.zeros(C, C, dtype=torch.long)
    for t, p in zip(y_va.tolist(), val_pred.tolist()):
        cm[t, p] += 1
    recall_per = []
    for c in range(C):
        support = cm[c].sum().item()
        correct = cm[c, c].item()
        recall_per.append({"view": view_ids[c], "support": support,
                           "recall": float(correct / support) if support > 0 else 0.0})
    return {
        "n_classes": C, "n_train": int(train_mask.sum()), "n_val": int(val_mask.sum()),
        "val_acc": float(val_acc), "best_val_acc": best_val,
        "majority_baseline": float(max(cm.sum(dim=0)).item() / cm.sum().item()) if cm.sum() > 0 else 0.0,
        "per_class_recall": recall_per,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--feature-mode",
        choices=[
            "encoder_pool",
            "z_shared",
            "z_phase",
            "z_view",
            "concat_shared_phase",
            "concat_all",
        ],
        default="encoder_pool",
    )
    parser.add_argument("--source-views", default="A4C")
    parser.add_argument("--target-views", default="A2C,A5C,A3C,PLAX,PSAX-MV")
    parser.add_argument("--num-studies", type=int, default=200)
    parser.add_argument("--phase-annotations", type=Path, default=Path("/opt/dlami/nvme/probe/phase_annotations.parquet"))
    parser.add_argument("--view-labels", type=Path, default=Path("/opt/dlami/nvme/data/view_labels/mimic_view_predictions.csv"))
    parser.add_argument("--splits-csv", type=Path, default=REPO_ROOT / "classifier" / "phase" / "splits" / "dicoms_split.csv")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--retrieval-mode",
        choices=["same_study_same_view", "same_study_any_view"],
        default="same_study_same_view",
        help=(
            "same_study_same_view (default): positive is same-study same-target-view, "
            "negatives are other-study same-target-view distractors. "
            "same_study_any_view: positive is same-study same-target-view, negatives are "
            "ALL other-study clips regardless of target view. If pilot 655 beats controls "
            "in same_study_any_view by a larger margin than in same_study_same_view, the "
            "learned signal is study-identity, not view-specific physiology."
        ),
    )
    parser.add_argument(
        "--slot-geometry",
        action="store_true",
        help="Emit held-out cosine statistics for same-study/other-study × same-view/cross-view pairs.",
    )
    parser.add_argument(
        "--view-classifier",
        action="store_true",
        help=(
            "Train a 1-layer linear probe on the encoded features to predict target_view. "
            "Uses 80/20 subject-disjoint split within the test set; reports val accuracy. "
            "Catches the 'pilot learned a view classifier' failure mode."
        ),
    )
    args = parser.parse_args()

    src_views = [v.strip() for v in args.source_views.split(",")]
    tgt_views = [v.strip() for v in args.target_views.split(",")]

    encoder, head, embed_dim = load_encoder_and_head(
        args.checkpoint, args.feature_mode,
    )

    # view_pair_policy: sampler uses np.choice(..., replace=False) across
    # 3 classes so all probs must be > 0. We down-weight same_view to near-
    # zero and weight cross-view heavily; target_view_sampling below
    # enforces the cross-view constraint anyway via require_different_view.
    view_pair_policy = {
        "enabled": True,
        "same_view_prob": 0.05,
        "same_family_prob": 0.35,
        "cross_family_prob": 0.60,
        "require_different_dicom": True,
        "allow_same_view": True,
        "resample_attempts": 8,
    }
    mv2sv_cfg = {
        "enabled": True,
        "target_view_sampling": {
            "stage": "stage2",
            "a4c_source_targets": [v for v in tgt_views if v in _APICAL],
            "allowed_targets": tgt_views,
            "target_dropout": 0.0,
            "require_different_view": True,
        },
        "fused_pool": {"enabled": False, "n_fused_min": 2, "n_fused_max": 2},
    }
    test_studies = _load_test_study_ids(args.splits_csv)
    print(f"[splits] test split has {len(test_studies)} unique studies")
    sampler = build_sampler(
        args.phase_annotations,
        mv2sv_cfg,
        view_pair_policy,
        args.view_labels,
        pairs_per_study=4,
        seed=args.seed,
    )

    # Draw records, then post-filter to test-split studies only.
    sampler.epoch = 0
    all_records = sampler.build_records()
    # Map row_idx -> s3_uri for URI lookup (internal attr is _df on the sampler).
    _sdf = getattr(sampler, "_df", None)
    if _sdf is None:
        _sdf = getattr(sampler, "df", None)
    row_to_uri = dict(zip(_sdf.index.astype(int), _sdf["s3_uri"].astype(str)))
    print(f"[sampler] drew {len(all_records)} records (unfiltered)")
    kept = []
    for r in all_records:
        if r.target_clip is None or r.target_view is None:
            continue
        if str(r.study_id) not in test_studies:
            continue
        sv = getattr(r.clip_a, "view", None)
        if sv is None or sv not in src_views:
            continue
        if r.target_view not in tgt_views:
            continue
        kept.append(r)
        if len(kept) >= args.num_studies:
            break
    print(f"[sampler] kept {len(kept)} records with source∈{src_views}, target∈{tgt_views}")
    if not kept:
        print("[ERROR] no records matched; nothing to evaluate")
        return 1

    # Encode each source + target clip
    def _uri_for(clip) -> str:
        raw = row_to_uri.get(int(clip.row_idx), "")
        if raw.startswith("s3://echodata25/mimic-raw-staging"):
            raw = raw.replace("s3://echodata25/mimic-raw-staging", "s3://echodata25/mimic-echo-224px", 1)
            if raw.endswith(".dcm"):
                raw = raw[:-4] + ".mp4"
        return raw

    def _frame_window(clip, frames_per_clip: int = 16, frame_step: int = 1) -> list[int]:
        # Centered window on anchor; clamp to [0, n_frames-1] happens inside _load_clip.
        start = int(clip.anchor_frame) - (frames_per_clip // 2) * frame_step
        return [max(0, start + i * frame_step) for i in range(frames_per_clip)]

    sources = []
    for k, r in enumerate(kept):
        try:
            src_uri = _uri_for(r.clip_a)
            src_idx = _frame_window(r.clip_a, r.frames_per_clip, r.frame_step)
            tgt_uri = _uri_for(r.target_clip)
            tgt_idx = _frame_window(r.target_clip, r.frames_per_clip, r.frame_step)
            if not src_uri or not tgt_uri:
                raise RuntimeError(f"missing URI for row_idx src={r.clip_a.row_idx} tgt={r.target_clip.row_idx}")
            src_clip = _load_clip(src_uri, src_idx)
            tgt_clip = _load_clip(tgt_uri, tgt_idx)
            src_feat = encode_clip(src_clip, encoder, head, args.feature_mode)
            tgt_feat = encode_clip(tgt_clip, encoder, head, args.feature_mode)
            sources.append({
                "study_id": r.study_id,
                "source_view": r.clip_a.view,
                "target_view": r.target_view,
                "source_feat": src_feat,
                "target_feat": tgt_feat,
            })
        except Exception as e:
            print(f"[warn] skipping row {k}: {e}")
            continue
        if (k + 1) % 25 == 0:
            print(f"  encoded {k+1}/{len(kept)}")

    print(f"[encode] successfully encoded {len(sources)} pairs")

    metrics = compute_retrieval_metrics(sources, mode=args.retrieval_mode)
    if args.slot_geometry:
        metrics["slot_geometry"] = compute_slot_geometry(sources)
    if args.view_classifier:
        metrics["view_classifier"] = train_view_classifier(sources)
    metrics["meta"] = {
        "checkpoint": str(args.checkpoint),
        "feature_mode": args.feature_mode,
        "retrieval_mode": args.retrieval_mode,
        "source_views": src_views,
        "target_views": tgt_views,
        "num_studies_drawn": args.num_studies,
        "num_encoded": len(sources),
        "slot_geometry": bool(args.slot_geometry),
        "view_classifier": bool(args.view_classifier),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[out] wrote {args.out}")
    if "overall" in metrics:
        o = metrics["overall"]
        print(f"[summary] n={o['n_rows']} top1={o['top1']:.3f} top5={o['top5']:.3f} gap={o['gap']:+.4f} neg_mean={o['valid_neg_count_mean']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
