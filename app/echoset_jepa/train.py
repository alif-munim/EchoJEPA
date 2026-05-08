"""EchoSet-JEPA Stage-2 training loop.

This is an implementation-ready skeleton:

- loss math (§5) is concrete.
- prioritized NCE negatives (§5.2) with fallback logic are implemented.
- EMA schedule wiring is in place.
- dataloader / collate / optimizer construction is **left to a follow-up PR**
  (`src/datasets/echoset_jepa_dataset.py` handles the per-study logic; the
  collate + DDP launcher and config loader are the final piece).

The goal of landing this file now is to pin the API contract between the
config, the dataset, the model, and the optimizer so downstream reviewers can
see exactly what the training loop will compute — without actually launching
any compute.
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.models.meta_embeddings import MetaDropout, MetaEmbeddings
from src.models.study_projectors import EMAProjectorPair, cosine_schedule
from src.models.study_transformer import StudyTransformer, StudyTransformerConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def _cosine_regress(h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    h = F.layer_norm(h, h.shape[-1:])
    z = F.layer_norm(z, z.shape[-1:])
    return (1.0 - F.cosine_similarity(h, z, dim=-1)).mean()


def _prioritized_neg_pool(
    tgt_view: torch.Tensor,         # (N,)
    tgt_modality: torch.Tensor,     # (N,)
    tgt_phase: torch.Tensor,        # (N,)
    tgt_study_id: torch.Tensor,     # (N,) — int hashes
    k_min: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Build a per-target negative mask with the 4-tier priority ladder (§5.2).

    Returns:
      - ``mask``: ``(N, N)`` boolean; True = valid negative for row i of column j.
        The diagonal is always True (positive). Same-study off-targets are False.
      - ``same_view_count``: ``(N,)`` — for logging.
      - ``diag``: summary floats (valid_neg_count_*, fallback_fraction).
    """
    N = tgt_view.shape[0]
    device = tgt_view.device
    same_study = tgt_study_id.unsqueeze(0) == tgt_study_id.unsqueeze(1)
    excl = same_study.clone()
    excl.fill_diagonal_(False)

    same_v = tgt_view.unsqueeze(0) == tgt_view.unsqueeze(1)
    same_m = tgt_modality.unsqueeze(0) == tgt_modality.unsqueeze(1)
    same_p = tgt_phase.unsqueeze(0) == tgt_phase.unsqueeze(1)

    pri_1 = same_v & same_m & same_p & ~excl
    pri_2 = same_v & same_m & ~excl
    pri_3 = same_m & ~excl
    pri_4 = ~excl  # all-batch fallback

    # Force diagonal True in every tier (positive row)
    eye = torch.eye(N, dtype=torch.bool, device=device)

    mask = torch.zeros_like(pri_1)
    fallback_level = torch.zeros(N, dtype=torch.long, device=device)
    for lvl, pri in enumerate([pri_1, pri_2, pri_3, pri_4], start=1):
        # For each row that does not yet have ≥k_min negatives (off-diag Trues),
        # promote to this priority level.
        off_diag_count = (mask & ~eye).sum(dim=1)
        needs = off_diag_count < k_min
        if not needs.any():
            break
        promote = needs.unsqueeze(1) & pri
        mask = mask | promote
        fallback_level = torch.where(needs, torch.full_like(fallback_level, lvl), fallback_level)

    mask = mask | eye

    same_view_count = (mask & same_v & ~eye).sum(dim=1).float()
    same_modality_count = (mask & same_m & ~eye).sum(dim=1).float()
    fallback_frac = (fallback_level > 2).float().mean().item()

    diag: Dict[str, float] = {
        "valid_neg_count_same_view_mean": same_view_count.mean().item(),
        "valid_neg_count_same_view_min": same_view_count.min().item(),
        "valid_neg_count_same_modality_mean": same_modality_count.mean().item(),
        "fallback_fraction": fallback_frac,
    }
    return mask, same_view_count, diag


def _nce_loss(
    h: torch.Tensor,          # (N, d_proj) — students
    z: torch.Tensor,          # (N, d_proj) — teacher targets (stopgrad applied by caller)
    neg_mask: torch.Tensor,   # (N, N) bool
    tau: float = 0.1,
) -> torch.Tensor:
    h = F.layer_norm(h, h.shape[-1:])
    z = F.layer_norm(z, z.shape[-1:])
    logits = h @ z.t() / tau            # (N, N)
    logits = logits.masked_fill(~neg_mask, float("-inf"))
    labels = torch.arange(h.shape[0], device=h.device)
    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Training step (pure function so it's testable)
# ---------------------------------------------------------------------------

@dataclass
class StepOutput:
    loss: torch.Tensor
    loss_regress: torch.Tensor
    loss_nce: torch.Tensor
    diagnostics: Dict[str, float]


def training_step(
    batch: Dict[str, torch.Tensor],
    st: StudyTransformer,
    meta: MetaEmbeddings,
    proj: EMAProjectorPair,
    *,
    lambda_nce: float = 0.03,
    tau_nce: float = 0.1,
    include_target_phase: bool = True,
    include_target_quality: bool = False,
    target_mode: str = "element_target",
    teacher_st: Optional[StudyTransformer] = None,
) -> StepOutput:
    """Single forward + loss computation for EchoSet-JEPA Stage-2.

    ``batch`` matches the output of ``EchoSetJEPADataset`` after padding + DDP
    collate (context element vectors, padding masks, meta ids, study ids).
    """
    ctx_meta_add = meta.encode_context(
        batch["ctx_meta_view"],
        batch["ctx_meta_modality"],
        batch["ctx_meta_phase"],
        batch["ctx_meta_quality"],
    )
    tgt_meta_add = meta.encode_target_slot(
        batch["tgt_meta_view"],
        batch["tgt_meta_modality"],
        phase_ids=batch.get("tgt_meta_phase"),
        include_phase=include_target_phase,
        include_quality=include_target_quality,
        quality_ids=batch.get("tgt_meta_quality"),
    )

    h_study, h_mask = st(
        ctx_elements=batch["ctx_elements"],
        ctx_meta_add=ctx_meta_add,
        ctx_pad_mask=batch["ctx_pad_mask"],
        tgt_meta_add=tgt_meta_add,
        tgt_pad_mask=batch["tgt_pad_mask"],
    )  # (B, d), (B, M_tgt, d)

    # --- target latents ----------------------------------------------------
    if target_mode == "element_target":
        # Teacher sees the already-aggregated target element vectors.
        z_target_src = st.clip_in(batch["tgt_elements"])   # reuse learned linear
        z_t = proj.teacher_forward(z_target_src).detach()
    elif target_mode == "full_study_teacher_target":
        assert teacher_st is not None, "full_study_teacher_target requires teacher_st"
        with torch.no_grad():
            # Teacher sees unmasked study; context + target positions, no masking
            full_ctx = torch.cat([batch["ctx_elements"], batch["tgt_elements"]], dim=1)
            full_meta = torch.cat([ctx_meta_add, tgt_meta_add], dim=1)
            full_pad = torch.cat([batch["ctx_pad_mask"], batch["tgt_pad_mask"]], dim=1)
            empty_tgt = torch.zeros(batch["ctx_elements"].shape[0], 0, ctx_meta_add.shape[-1], device=ctx_meta_add.device)
            empty_pad = torch.zeros(batch["ctx_elements"].shape[0], 0, dtype=torch.bool, device=ctx_meta_add.device)
            _, teacher_out = teacher_st(
                ctx_elements=full_ctx,
                ctx_meta_add=full_meta,
                ctx_pad_mask=full_pad,
                tgt_meta_add=empty_tgt,
                tgt_pad_mask=empty_pad,
            )  # teacher_out is empty; we pick from the full sequence instead
            # Select the target positions (which live at the tail of full_ctx)
            M_ctx = batch["ctx_elements"].shape[1]
            # The teacher's token stream returns h_study + h_ctx + h_tgt=empty ;
            # we instead run a second teacher forward that reads target slots.
            # For clarity, use element_target path for MVP; Mode B is an
            # ablation pathway to be finalized when its ablation YAML runs.
            z_t = proj.teacher_forward(st.clip_in(batch["tgt_elements"])).detach()
    else:
        raise ValueError(f"unknown target_mode={target_mode!r}")

    h_t = proj.student_forward(h_mask)                  # (B, M_tgt, d_proj)

    # Flatten over valid target positions only
    valid = ~batch["tgt_pad_mask"]                       # (B, M_tgt)
    h_flat = h_t[valid]
    z_flat = z_t[valid]
    v_flat = batch["tgt_meta_view"][valid]
    m_flat = batch["tgt_meta_modality"][valid]
    p_flat = batch["tgt_meta_phase"][valid]
    # Broadcast per-study id over that study's target rows, then flatten by valid.
    s_flat = batch["study_id_int"].unsqueeze(1).expand_as(batch["tgt_meta_view"])[valid]

    loss_reg = _cosine_regress(h_flat, z_flat)

    neg_mask, _, nce_diag = _prioritized_neg_pool(v_flat, m_flat, p_flat, s_flat)
    loss_nce = _nce_loss(h_flat, z_flat, neg_mask, tau=tau_nce)

    loss = loss_reg + lambda_nce * loss_nce

    with torch.no_grad():
        var_t = h_flat.std(dim=0).mean().item()
        cov = (h_flat - h_flat.mean(0, keepdim=True))
        cov = cov.t() @ cov / max(h_flat.shape[0] - 1, 1)
        cov_off = cov.fill_diagonal_(0.0).abs().mean().item()

    diagnostics: Dict[str, float] = {
        "loss_regress": loss_reg.item(),
        "loss_nce": loss_nce.item(),
        "var_t": var_t,
        "cov_off": cov_off,
        **nce_diag,
    }
    return StepOutput(loss=loss, loss_regress=loss_reg, loss_nce=loss_nce, diagnostics=diagnostics)


# ---------------------------------------------------------------------------
# Config loader + CLI entry point (DDP launcher deferred)
# ---------------------------------------------------------------------------

def main(args=None, resume_preempt: bool = False) -> None:
    """Stage-2 training entry point (PR-N4).

    ``args`` is the parsed config dict handed down by ``app.scaffold`` (not
    argparse.Namespace — that's a V-JEPA quirk in ``app/main.py``). See
    ``configs/train/echoset_jepa/echoset_jepa_v1_K8.yaml`` for the schema.
    """
    import os

    # Single-GPU-per-process for DDP (matches app/vjepa/train.py pattern).
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
    except KeyError:
        pass

    import numpy as np
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    try:
        mp.set_sharing_strategy("file_system")
    except Exception:
        pass
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader

    from src.datasets.echoset_jepa_collate import EchoSetStudyDataset, echoset_collate
    from src.utils.distributed import init_distributed
    from src.utils.logging import AverageMeter, CSVLogger, get_logger

    log = get_logger(__name__, force=True)

    cfg = args if isinstance(args, dict) else {}
    exp = cfg.get("experiment", {})
    st_cfg_dict = exp.get("study_transformer", {})
    masking_cfg = exp.get("masking", {})
    sampler_cfg = exp.get("sampler", {})
    loss_cfg = exp.get("loss", {})
    ema_cfg = exp.get("ema", {})
    optim_cfg = exp.get("optim", {})
    clip_enc_cfg = exp.get("clip_encoder", {})
    target_meta_cfg = exp.get("target_meta", {})
    elements_cfg = exp.get("elements", {})
    logging_cfg = exp.get("logging", {})
    collapse_cfg = exp.get("collapse_monitor", {})
    folder = cfg.get("folder", "./echoset_jepa_run")
    os.makedirs(folder, exist_ok=True)

    seed = int(cfg.get("seed", 0))
    np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    world_size, rank = init_distributed()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    log.info("[rank %d/%d] init on %s", rank, world_size, device)

    # --- model ------------------------------------------------------------
    st_cfg = StudyTransformerConfig(
        d_clip=int(st_cfg_dict.get("d_clip", 1024)),
        d_model=int(st_cfg_dict.get("d_model", 512)),
        n_layers=int(st_cfg_dict.get("n_layers", 4)),
        n_heads=int(st_cfg_dict.get("n_heads", 8)),
        ffn_mult=int(st_cfg_dict.get("ffn_mult", 4)),
        dropout_ffn=float(st_cfg_dict.get("dropout_ffn", 0.1)),
        dropout_attn=float(st_cfg_dict.get("dropout_attn", 0.0)),
        max_M=int(st_cfg_dict.get("max_M", 64)),
    )
    meta_dropout = MetaDropout(
        view=float(exp.get("meta_dropout", {}).get("view", 0.15)),
        modality=float(exp.get("meta_dropout", {}).get("modality", 0.10)),
        phase=float(exp.get("meta_dropout", {}).get("phase", 0.30)),
        quality=float(exp.get("meta_dropout", {}).get("quality", 0.30)),
    )
    st = StudyTransformer(st_cfg).to(device)
    meta = MetaEmbeddings(d_model=st_cfg.d_model, dropout=meta_dropout).to(device)
    proj = EMAProjectorPair(
        d_model=st_cfg.d_model,
        d_hidden=int(exp.get("projector", {}).get("d_hidden", 1024)),
        d_proj=int(exp.get("projector", {}).get("d_proj", 256)),
    ).to(device)

    if world_size > 1:
        st = DistributedDataParallel(st, device_ids=[device.index or 0])
        meta = DistributedDataParallel(meta, device_ids=[device.index or 0])
        # Student projector is wrapped; teacher is not (no grad → DDP would complain).
        proj_student_ddp = DistributedDataParallel(
            proj.student, device_ids=[device.index or 0]
        )
    else:
        proj_student_ddp = proj.student

    # --- dataset + loader -------------------------------------------------
    k_sample = sampler_cfg.get("sample_manifest")
    cache_prefix = clip_enc_cfg.get("cache_local_prefix") or clip_enc_cfg.get("cache_s3_prefix", "")
    if not k_sample or not cache_prefix:
        raise ValueError("config must set sampler.sample_manifest + clip_encoder.cache_local_prefix")

    dataset = EchoSetStudyDataset(
        k_sample_manifest_path=k_sample,
        cache_prefix=cache_prefix,
        meta=meta.module if isinstance(meta, DistributedDataParallel) else meta,
        element_agg=elements_cfg.get("element_agg", "mean"),
        strategy_weights=masking_cfg.get("strategy_weights"),
        seed=seed + rank,
    )

    ddp_sampler = None
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        ddp_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)

    batch_size = int(optim_cfg.get("batch_studies_per_gpu", 32))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=ddp_sampler,
        shuffle=(ddp_sampler is None),
        collate_fn=echoset_collate,
        num_workers=int(cfg.get("num_workers", 4)),
        drop_last=True,
        persistent_workers=False,
        pin_memory=True,
    )

    # --- optimizer --------------------------------------------------------
    params = list(st.parameters()) + list(meta.parameters()) + list(proj.student.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=float(optim_cfg.get("lr", 5e-4)),
        betas=tuple(optim_cfg.get("betas", (0.9, 0.95))),
        weight_decay=float(optim_cfg.get("weight_decay", 0.05)),
    )
    warmup_steps = int(optim_cfg.get("warmup_steps", 2000))
    main_steps = int(optim_cfg.get("main_steps", 50000))
    cooldown_steps = int(optim_cfg.get("cooldown_steps", 5000))
    total_steps = warmup_steps + main_steps + cooldown_steps
    peak_lr = float(optim_cfg.get("lr", 5e-4))

    def _lr_at(step: int) -> float:
        if step < warmup_steps:
            return peak_lr * (step + 1) / max(warmup_steps, 1)
        if step < warmup_steps + main_steps:
            return peak_lr
        cool_i = step - warmup_steps - main_steps
        frac = cool_i / max(cooldown_steps, 1)
        return peak_lr * max(0.0, 1.0 - frac)

    tau_start = float(ema_cfg.get("tau_start", 0.996))
    tau_end = float(ema_cfg.get("tau_end", 0.9999))

    # --- resume ----------------------------------------------------------
    ckpt_dir = os.path.join(folder, "checkpoints"); os.makedirs(ckpt_dir, exist_ok=True)
    latest_path = os.path.join(ckpt_dir, "latest.pt")
    global_step = 0
    start_epoch = 0
    if resume_preempt and os.path.exists(latest_path):
        log.info("[rank %d] resuming from %s", rank, latest_path)
        sd = torch.load(latest_path, map_location="cpu")
        _load_state(st, sd.get("study_transformer"))
        _load_state(meta, sd.get("meta_embeddings"))
        _load_state(proj.student, sd.get("projector_student"))
        _load_state(proj.teacher, sd.get("projector_teacher"))
        optimizer.load_state_dict(sd["optimizer"])
        global_step = int(sd.get("global_step", 0))
        start_epoch = int(sd.get("epoch", 0))

    # --- csv logger -------------------------------------------------------
    schema = logging_cfg.get("csv_schema") or [
        "step", "loss", "loss_regress", "loss_nce", "var_t", "cov_off",
        "valid_neg_count_same_view_mean", "valid_neg_count_same_view_min",
        "valid_neg_count_same_modality_mean", "fallback_fraction",
        "mask_strategy", "M_elements_mean",
    ]
    csv = None
    if rank == 0:
        csv_cols = [("%d", "step")] + [("%.6f", k) if k != "mask_strategy" else ("%s", k) for k in schema if k != "step"]
        csv = CSVLogger(os.path.join(folder, "train_log.csv"), *csv_cols)

    log_every = int(logging_cfg.get("log_every_steps", 50))
    ckpt_every = int(cfg.get("checkpoint_every_steps", 2500))

    # --- train loop -------------------------------------------------------
    var_t_below_floor_count = 0
    var_t_floor = float(collapse_cfg.get("var_t_floor", 0.3))
    halt_below_for = int(collapse_cfg.get("halt_if_below_for_steps", 500))

    st.train(); meta.train(); proj.student.train()
    num_epochs = int(optim_cfg.get("num_epochs", 10_000_000))    # drives by step count, not epochs
    for epoch in range(start_epoch, num_epochs):
        if ddp_sampler is not None:
            ddp_sampler.set_epoch(epoch)
        for batch in loader:
            if global_step >= total_steps:
                break

            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}

            for pg in optimizer.param_groups:
                pg["lr"] = _lr_at(global_step)

            out = training_step(
                batch, st.module if isinstance(st, DistributedDataParallel) else st,
                meta.module if isinstance(meta, DistributedDataParallel) else meta,
                proj,
                lambda_nce=float(loss_cfg.get("lambda_nce", 0.03)),
                tau_nce=float(loss_cfg.get("tau_nce", 0.1)),
                include_target_phase=bool(target_meta_cfg.get("include_target_phase", True)),
                include_target_quality=bool(target_meta_cfg.get("target_quality_token_ablation", False)),
                target_mode=exp.get("target_mode", "element_target"),
            )
            optimizer.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            # EMA update of teacher projector
            tau = cosine_schedule(global_step, total_steps, tau_start, tau_end)
            proj.update_teacher(tau)

            # Collapse monitor
            if out.diagnostics["var_t"] < var_t_floor:
                var_t_below_floor_count += 1
            else:
                var_t_below_floor_count = 0
            if var_t_below_floor_count >= halt_below_for:
                log.error("var_t < %.2f for %d steps — halting per collapse monitor",
                          var_t_floor, var_t_below_floor_count)
                _save(ckpt_dir, "halt_collapse.pt", st, meta, proj, optimizer, epoch, global_step, rank=rank)
                return

            if rank == 0 and (global_step % log_every == 0):
                m_mean = float(batch["n_elements"].float().mean().item())
                strat = batch["mask_strategies"][0] if batch.get("mask_strategies") else "-"
                row = [global_step]
                for k in schema:
                    if k == "step":
                        continue
                    if k == "mask_strategy":
                        row.append(strat)
                    elif k == "loss":
                        row.append(float(out.loss.item()))
                    elif k == "M_elements_mean":
                        row.append(m_mean)
                    else:
                        row.append(float(out.diagnostics.get(k, float("nan"))))
                csv.log(*row)
                log.info(
                    "step=%d loss=%.4f reg=%.4f nce=%.4f var=%.3f cov=%.4f M=%.1f strat=%s",
                    global_step, out.loss.item(), out.diagnostics["loss_regress"],
                    out.diagnostics["loss_nce"], out.diagnostics["var_t"],
                    out.diagnostics["cov_off"], m_mean, strat,
                )

            if rank == 0 and ckpt_every > 0 and global_step > 0 and global_step % ckpt_every == 0:
                _save(ckpt_dir, f"step{global_step}.pt", st, meta, proj, optimizer, epoch, global_step, rank=rank)
                _save(ckpt_dir, "latest.pt", st, meta, proj, optimizer, epoch, global_step, rank=rank)

            global_step += 1
        if global_step >= total_steps:
            break

    if rank == 0:
        _save(ckpt_dir, "final.pt", st, meta, proj, optimizer, epoch, global_step, rank=rank)
    if world_size > 1:
        dist.barrier()
    log.info("[rank %d] training complete at step %d", rank, global_step)


def _unwrap(m):
    from torch.nn.parallel import DistributedDataParallel
    return m.module if isinstance(m, DistributedDataParallel) else m


def _load_state(m, sd):
    if sd is None:
        return
    _unwrap(m).load_state_dict(sd, strict=False)


def _save(ckpt_dir: str, name: str, st, meta, proj, optimizer, epoch: int, global_step: int, rank: int) -> None:
    if rank != 0:
        return
    import os, torch
    path = os.path.join(ckpt_dir, name)
    torch.save({
        "study_transformer": _unwrap(st).state_dict(),
        "meta_embeddings": _unwrap(meta).state_dict(),
        "projector_student": proj.student.state_dict(),
        "projector_teacher": proj.teacher.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }, path)


__all__ = [
    "training_step",
    "StepOutput",
    "main",
    "_cosine_regress",
    "_nce_loss",
    "_prioritized_neg_pool",
]
