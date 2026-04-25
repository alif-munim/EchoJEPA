"""Pre-pool per-tubelet feature extraction for diff-probe analysis.

Drops in as an `eval_name` module compatible with `evals.main`. Loads the
frozen encoder exactly as video_classification_frozen would (same
ClipAggregation / VideoMAEWrapper adapters, same dataloader, same
num_segments=2 logic), then caches the pre-pool token tensor for every
test clip, spatially averaged to `[num_segments=2, T=8, D]` per clip.

See `claude/neurips/experiments/diff-probe-analysis.md` for the protocol.
"""

import logging
import os
import tempfile

import numpy as np
import torch
import torch.multiprocessing as mp

from evals.video_classification_frozen.eval import make_dataloader
from evals.video_classification_frozen.models import init_module as init_encoder
from src.utils.distributed import init_distributed


def _patch_videomae_prepool(encoder):
    """Monkey-patch VideoMAEWrapper's inner model.forward_features to return
    pre-pool tokens instead of the global-mean-pooled [B, D] vector.

    The vendored VideoMAE modeling_finetune.VisionTransformer.forward_features
    ends with `self.fc_norm(x.mean(1))`, yielding a pooled [B, D] that the
    wrapper lifts to [B, 1, D]. For pre-pool extraction we need [B, T*S, D]
    post-blocks. We bypass `.mean(1)` and `fc_norm` to preserve the full
    tubelet sequence (fc_norm is a mean-pool-specific LN that would be a
    shape mismatch anyway).
    """
    # encoder is either ClipAggregation(wrapper) or VideoMAEWrapper directly.
    wrapper = encoder
    while hasattr(wrapper, "backbone") and not hasattr(wrapper, "model"):
        wrapper = wrapper.backbone
    # ClipAggregation stores the underlying wrapper in ``module`` in some
    # codepaths; fall back to the eval-init signature we know works.
    model = getattr(wrapper, "model", None)
    if model is None or not hasattr(model, "forward_features"):
        return False  # not a VideoMAEWrapper

    # Only patch if this is the vendored VideoMAE finetune VisionTransformer
    if not (hasattr(model, "patch_embed") and hasattr(model, "blocks") and
            hasattr(model, "pos_drop")):
        return False

    import types

    @torch.no_grad()
    def forward_features_prepool(self, x):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            pe = self.pos_embed.expand(x.shape[0], -1, -1).type_as(x).to(x.device)
            pe = self._get_matched_pos_embed(pe, x.size(1))
            x = x + pe
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x  # [B, T*S, D] pre-pool

    model.forward_features = types.MethodType(forward_features_prepool, model)
    logger.info("Patched VideoMAE forward_features for pre-pool extraction.")
    return True

# AF_UNIX length guard (same as video_classification_frozen.eval)
_short_tmp = "/tmp/vjepa_run"
os.makedirs(_short_tmp, exist_ok=True)
tempfile.tempdir = _short_tmp
os.environ["TMPDIR"] = _short_tmp

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def _reshape_to_segments(out_list, num_segments, num_tubelets_t, num_spatial):
    """Normalize adapter output to [B, num_segments, T, S, D].

    ``ClipAggregation`` (JEPA/BYOL/SALT adapter) returns a list of length
    ``num_views_per_segment`` (always 1 in our configs). The single tensor
    is [B, num_segments * T * S, D].

    ``VideoMAEWrapper`` (MAE adapter) returns a list of length
    ``num_segments`` with each item [B, T * S, D].
    """
    if len(out_list) == 1:
        x = out_list[0]
        B, N, D = x.shape
        expected = num_segments * num_tubelets_t * num_spatial
        if N != expected:
            raise RuntimeError(
                f"Unexpected token count {N}, expected {expected} "
                f"(num_segments={num_segments}, T={num_tubelets_t}, S={num_spatial})"
            )
        return x.reshape(B, num_segments, num_tubelets_t, num_spatial, D)

    if len(out_list) == num_segments:
        B, N, D = out_list[0].shape
        expected = num_tubelets_t * num_spatial
        if N != expected:
            raise RuntimeError(
                f"Per-segment token count {N}, expected {expected} "
                f"(T={num_tubelets_t}, S={num_spatial})"
            )
        stacked = torch.stack(out_list, dim=1)  # [B, num_segments, N, D]
        return stacked.reshape(B, num_segments, num_tubelets_t, num_spatial, D)

    raise RuntimeError(
        f"Unexpected encoder output list length {len(out_list)}; "
        f"expected 1 (ClipAggregation) or num_segments={num_segments} (VideoMAEWrapper)"
    )


def main(args_eval, resume_preempt=False):
    # ---- Config parsing (mirrors video_classification_frozen.eval.main) ----
    import os
    from src.utils.distributed import init_distributed  # noqa: F811

    def _override(env_var, cfg, key, cast=str):
        v = os.environ.get(env_var)
        if v is None:
            return
        if cast is bool:
            v = v.lower() in ("true", "1", "yes", "t")
        else:
            v = cast(v)
        logger.info(f"!!! MANUAL OVERRIDE: {key} -> {v}")
        cfg[key] = v

    _override("OVERRIDE_TAG", args_eval, "tag")
    _override("OVERRIDE_FOLDER", args_eval, "folder")
    exp = args_eval.setdefault("experiment", {})
    data = exp.setdefault("data", {})
    opt = exp.setdefault("optimization", {})
    _override("OVERRIDE_VAL_DATA", data, "dataset_val")
    _override("OVERRIDE_BATCH", opt, "batch_size", int)
    # Checkpoint / model-name overrides let one YAML cover many ckpts.
    mk = args_eval.setdefault("model_kwargs", {})
    _override("OVERRIDE_CKPT", mk, "checkpoint")
    _override("OVERRIDE_MODULE_NAME", mk, "module_name")
    enc_cfg = mk.setdefault("pretrain_kwargs", {}).setdefault("encoder", {})
    _override("OVERRIDE_MODEL_NAME", enc_cfg, "model_name")
    # Where to write the cache (NVMe)
    output_path = os.environ.get("FEATURES_OUTPUT", None)
    if output_path is None:
        output_path = os.path.join(args_eval["folder"], "features.pt")
    logger.info(f"Will write features to: {output_path}")

    # ---- Model / data config fields ----
    pretrain = args_eval["model_kwargs"]
    checkpoint = pretrain["checkpoint"]
    module_name = pretrain["module_name"]
    model_kwargs = pretrain["pretrain_kwargs"]
    wrapper_kwargs = pretrain.get("wrapper_kwargs", {}) or {}

    d = args_eval["experiment"]["data"]
    dataset_type = d.get("dataset_type", "VideoDataset")
    val_csv = d["dataset_val"]
    resolution = d.get("resolution", 224)
    frames_per_clip = d.get("frames_per_clip", 16)
    frame_step = d.get("frame_step", 2)
    num_segments = d.get("num_segments", 2)
    num_views_per_segment = d.get("num_views_per_segment", 1)

    o = args_eval["experiment"]["optimization"]
    batch_size = o.get("batch_size", 8)
    use_bfloat16 = o.get("use_bfloat16", True)

    # ---- Distributed setup (inherited from evals.main process_main) ----
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for pre-pool extraction.")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info(f"rank {rank}/{world_size}")

    # ---- Build encoder (frozen) ----
    encoder = init_encoder(
        module_name=module_name,
        frames_per_clip=frames_per_clip,
        resolution=resolution,
        checkpoint=checkpoint,
        model_kwargs=model_kwargs,
        wrapper_kwargs=wrapper_kwargs,
        device=device,
    )
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # VideoMAE's default forward_features does a global-mean pool; patch to
    # return the [B, T*S, D] token sequence for pre-pool caching.
    _patch_videomae_prepool(encoder)

    # ---- Build dataloader (deterministic, no shuffle, matches job 216/220) ----
    val_loader, _ = make_dataloader(
        root_path=[val_csv],
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        dataset_type=dataset_type,
        img_size=resolution,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_segments=num_segments,
        num_views_per_segment=num_views_per_segment,
        allow_segment_overlap=True,
        training=False,
        num_workers=8,
    )

    # ---- Inferred dims (verified against vit_encoder_multiclip.py) ----
    tubelet_size = model_kwargs.get("encoder", {}).get("tubelet_size", 2)
    patch_size = model_kwargs.get("encoder", {}).get("patch_size", 16)
    T = frames_per_clip // tubelet_size
    spatial_per_side = resolution // patch_size
    S = spatial_per_side * spatial_per_side
    logger.info(f"Expected tubelet layout: T={T}, spatial={S} ({spatial_per_side}^2), num_segments={num_segments}")

    # ---- Extract ----
    from torch.amp import autocast
    from tqdm import tqdm

    local_chunks = {"features": [], "labels": [], "paths": []}
    iterator = tqdm(val_loader, desc=f"rank {rank} extract", unit="batch", dynamic_ncols=True, disable=(rank != 0))

    embed_dim = encoder.embed_dim
    total_clips = 0

    with torch.no_grad():
        for data in iterator:
            clips = [[dij.to(device, non_blocking=True) for dij in di] for di in data[0]]
            clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
            labels = data[1]
            paths = data[3] if len(data) > 3 else [f"unknown_{i}" for i in range(len(labels))]

            with autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
                outputs = encoder(clips, clip_indices)

            # Normalize to [B, num_segments, T, S, D]
            reshaped = _reshape_to_segments(outputs, num_segments, T, S)
            # Spatial mean (default) -> [B, num_segments, T, D], cast to fp16 for cache.
            # FEATURE_KEEP_SPATIAL=1 keeps the spatial axis -> [B, num_segments, T, S, D]
            # for the per-token diff-probe variant (caveat in diff-probe-analysis.md).
            if os.environ.get("FEATURE_KEEP_SPATIAL", "0") == "1":
                pooled = reshaped.half().cpu()
            else:
                pooled = reshaped.float().mean(dim=3).half().cpu()

            local_chunks["features"].append(pooled)
            local_chunks["labels"].append(labels.float().cpu())
            local_chunks["paths"].extend(list(paths))
            total_clips += pooled.shape[0]

    features = torch.cat(local_chunks["features"], dim=0) if local_chunks["features"] else torch.empty(0)
    labels = torch.cat(local_chunks["labels"], dim=0) if local_chunks["labels"] else torch.empty(0)
    logger.info(f"rank {rank}: extracted features shape {tuple(features.shape)} labels {tuple(labels.shape)}")

    # Each rank writes its own shard; driver script concatenates.
    shard_path = output_path.replace(".pt", f".rank{rank}.pt")
    os.makedirs(os.path.dirname(shard_path) or ".", exist_ok=True)
    torch.save(
        {
            "features": features,
            "labels": labels,
            "paths": local_chunks["paths"],
            "meta": {
                "checkpoint": checkpoint,
                "module_name": module_name,
                "frames_per_clip": frames_per_clip,
                "frame_step": frame_step,
                "num_segments": num_segments,
                "tubelet_T": T,
                "embed_dim": embed_dim,
                "rank": rank,
                "world_size": world_size,
            },
        },
        shard_path,
    )
    logger.info(f"rank {rank}: wrote {shard_path}")

    # Barrier via file presence — rank 0 concatenates shards after everyone is done.
    if world_size > 1:
        torch.distributed.barrier()

    if rank == 0:
        shards = sorted(
            output_path.replace(".pt", f".rank{r}.pt") for r in range(world_size)
        )
        all_feats, all_labs, all_paths = [], [], []
        meta = None
        for sp in shards:
            d_shard = torch.load(sp, map_location="cpu", weights_only=False)
            all_feats.append(d_shard["features"])
            all_labs.append(d_shard["labels"])
            all_paths.extend(d_shard["paths"])
            meta = d_shard["meta"]
        cat_feats = torch.cat(all_feats, dim=0)
        cat_labs = torch.cat(all_labs, dim=0)
        # DistributedSampler pads with duplicates so N % world_size == 0.
        # Dedupe on path (first occurrence wins), preserving input CSV order is
        # not required — downstream we join on path anyway.
        seen = {}
        keep = []
        for i, p in enumerate(all_paths):
            if p not in seen:
                seen[p] = i
                keep.append(i)
        keep_idx = torch.tensor(keep, dtype=torch.long)
        n_dupes = len(all_paths) - len(keep)
        if n_dupes > 0:
            logger.info(f"rank 0: deduplicated {n_dupes} padded clips")
        merged = {
            "features": cat_feats.index_select(0, keep_idx),
            "labels": cat_labs.index_select(0, keep_idx),
            "paths": [all_paths[i] for i in keep],
            "meta": meta,
        }
        torch.save(merged, output_path)
        logger.info(f"rank 0: merged {len(shards)} shards -> {output_path} "
                    f"shape={tuple(merged['features'].shape)} n_clips={len(merged['paths'])}")
        for sp in shards:
            try:
                os.remove(sp)
            except OSError:
                pass

    if world_size > 1:
        torch.distributed.barrier()
