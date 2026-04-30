# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from logging import getLogger
from multiprocessing import Value
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.masks.phase_mask_utils import (
    PhaseBucketSpec,
    PhaseMaskStats,
    apply_shuffled_hr,
    block_center_phi,
    block_center_time_tubelets,
    choose_bucket,
    circular_dist,
    cycle_tubelets_from_hr,
    dphi_fraction,
    parse_bucket_cfg,
    validate_hr,
)

_GLOBAL_SEED = 0
logger = getLogger()


class MaskCollator(object):

    def __init__(
        self,
        cfgs_mask,
        dataset_fpcs,
        crop_size=(224, 224),
        patch_size=(16, 16),
        tubelet_size=2,
        fps_sampled: Optional[float] = None,
        phase_mask_cfg: Optional[dict] = None,
    ):
        super(MaskCollator, self).__init__()

        self.mask_generators = dict()
        # Expose (D, H, W) grid dims per fpc so downstream code (phi-JEPA training
        # step) can unflatten mask indices without re-deriving the grid shape.
        self.grid_dims_per_fpc: dict = {}
        self.tubelet_size = int(tubelet_size)
        self.fps_sampled = float(fps_sampled) if fps_sampled is not None else None
        for fpc in dataset_fpcs:
            self.mask_generators[fpc] = []
            for m in cfgs_mask:
                mask_generator = _MaskGenerator(
                    crop_size=crop_size,
                    num_frames=fpc,
                    spatial_patch_size=patch_size,
                    temporal_patch_size=tubelet_size,
                    spatial_pred_mask_scale=m.get("spatial_scale"),
                    temporal_pred_mask_scale=m.get("temporal_scale"),
                    aspect_ratio=m.get("aspect_ratio"),
                    npred=m.get("num_blocks"),
                    max_context_frames_ratio=m.get("max_temporal_keep", 1.0),
                    max_keep=m.get("max_keep", None),
                    full_complement=m.get("full_complement", False),
                    pred_full_complement=m.get("pred_full_complement", False),
                    inv_block=m.get("inv_block", False),
                )
                self.mask_generators[fpc].append(mask_generator)
            if self.mask_generators[fpc]:
                gen = self.mask_generators[fpc][0]
                self.grid_dims_per_fpc[fpc] = (gen.duration, gen.height, gen.width)

        # -----------------------------------------------------------------
        # Phase-aware target sampling (phi-JEPA mask-phi variant).
        #
        # NOTE on unit convention (see src/masks/phase_mask_utils.py):
        #   DICOM FrameTime is the *native* acquisition time per frame, but
        #   the V-JEPA dataloader resamples each clip to a uniform `fps`
        #   before the encoder sees it. Therefore masking reasons about phase
        #   on the sampled tubelet grid using `cycle_tubelets_from_hr(...)`;
        #   `frame_time_ms` is used only as a metadata-validity check.
        # -----------------------------------------------------------------
        self.phase_aware = False
        self.phase_mask_cfg = phase_mask_cfg or {}
        if self.phase_mask_cfg.get("phase_aware", False):
            self.phase_aware = True
            if self.fps_sampled is None or self.fps_sampled <= 0:
                raise ValueError(
                    "phase_aware=True requires MaskCollator to be given fps_sampled"
                )
            self.phase_specs, self.phase_probs = parse_bucket_cfg(
                self.phase_mask_cfg.get("phase_buckets"),
                self.phase_mask_cfg.get("phase_bucket_probs"),
            )
            self.phase_fallback = str(self.phase_mask_cfg.get("phase_fallback", "random"))
            self.require_valid_hr = bool(self.phase_mask_cfg.get("require_valid_hr", True))
            self.shuffled_hr = bool(self.phase_mask_cfg.get("shuffled_hr", False))
            self.phase_max_attempts = int(self.phase_mask_cfg.get("phase_max_attempts", 20))
            self.phase_seed_base = int(self.phase_mask_cfg.get("phase_seed", 0))
            # Step counter for per-batch rng seeding (separate from the block-size
            # counter on _MaskGenerator so phase_aware=False is bit-identical).
            self._phase_itr = Value("i", -1)
            # PhaseMaskStats lives in shared memory so DataLoader worker
            # subprocess updates propagate back to the main process at
            # end-of-epoch. Bucket names must be registered up-front.
            self.stats = PhaseMaskStats(
                bucket_names=list(self.phase_probs.keys()),
            )
            logger.info(
                "MaskCollator: phase-aware target sampling ENABLED "
                f"(buckets={list(self.phase_probs.keys())}, "
                f"shuffled_hr={self.shuffled_hr}, fallback={self.phase_fallback})"
            )
            # Sanity warning: with temporal_scale=[1.0, 1.0] target blocks
            # span the full clip, leaving no room for phase-aware t_start
            # selection. Mask-phi becomes equivalent to the vanilla path.
            all_full_span = all(
                (float(m.get("temporal_scale", (1.0, 1.0))[0]) >= 1.0 - 1e-6)
                and (float(m.get("temporal_scale", (1.0, 1.0))[1]) >= 1.0 - 1e-6)
                for m in cfgs_mask
            )
            if all_full_span:
                logger.warning(
                    "phase-aware masking is ENABLED but every mask generator has "
                    "temporal_scale=[1.0, 1.0]; target blocks span the full clip, "
                    "so phase-gating has no degrees of freedom. Use a localized "
                    "temporal_scale (e.g. [0.25, 0.25]) for meaningful Mask-phi."
                )

    def _phase_step(self) -> int:
        i = self._phase_itr
        with i.get_lock():
            i.value += 1
            return i.value

    def step(self):
        for fpc in self.mask_generators:
            for mask_generator in self.mask_generators[fpc]:
                mask_generator.step()

    # ------------------------------------------------------------------
    # Vanilla path
    # ------------------------------------------------------------------

    def _vanilla_call(self, batch):
        filtered_batches = {fpc: [] for fpc in self.mask_generators}
        for sample in batch:
            fpc = 1  # default for images / stills
            if len(sample) >= 3:
                clip_indices = sample[2]
                if isinstance(clip_indices, (list, tuple)) and len(clip_indices) > 0:
                    last_clip = clip_indices[-1]
                    if isinstance(last_clip, (list, tuple)):
                        fpc = len(last_clip)
                    elif hasattr(last_clip, '__len__'):
                        fpc = len(last_clip)
            if fpc in filtered_batches:
                filtered_batches[fpc] += [sample]

        fpc_collations = []
        for fpc in filtered_batches:
            fpc_batch = filtered_batches[fpc]
            batch_size = len(fpc_batch)
            if batch_size == 0:
                continue
            collated_batch = torch.utils.data.default_collate(fpc_batch)
            collated_masks_pred, collated_masks_enc = [], []
            for i, mask_generator in enumerate(self.mask_generators[fpc]):
                masks_enc, masks_pred = mask_generator(batch_size)
                collated_masks_enc.append(masks_enc)
                collated_masks_pred.append(masks_pred)
            fpc_collations += [
                (collated_batch, collated_masks_enc, collated_masks_pred)
            ]

        return fpc_collations

    # ------------------------------------------------------------------
    # Phase-aware path
    # ------------------------------------------------------------------

    def _phase_call(self, batch):
        # Split samples by fpc.
        filtered_batches = {fpc: [] for fpc in self.mask_generators}
        for sample in batch:
            fpc = 1
            if len(sample) >= 3:
                clip_indices = sample[2]
                if isinstance(clip_indices, (list, tuple)) and len(clip_indices) > 0:
                    last_clip = clip_indices[-1]
                    if isinstance(last_clip, (list, tuple)):
                        fpc = len(last_clip)
                    elif hasattr(last_clip, '__len__'):
                        fpc = len(last_clip)
            if fpc in filtered_batches:
                filtered_batches[fpc] += [sample]

        seed = self._phase_step()
        rng = np.random.default_rng(self.phase_seed_base + seed)

        fpc_collations = []
        for fpc in filtered_batches:
            fpc_batch = filtered_batches[fpc]
            batch_size = len(fpc_batch)
            if batch_size == 0:
                continue
            D, H, W = self.grid_dims_per_fpc[fpc]
            HW = H * W

            # Extract per-sample metadata (meta dict is sample[4] when present).
            hr_list: List[float] = []
            ft_list: List[float] = []
            nf_list: List[int] = []
            for s in fpc_batch:
                meta = s[4] if len(s) >= 5 and isinstance(s[4], dict) else {}
                hr_list.append(float(meta.get("hr_bpm", float("nan"))))
                ft_list.append(float(meta.get("frame_time_ms", float("nan"))))
                # Derive native num_frames when absent; fpc is always present
                # on the sampled grid. Use sampled-grid num_frames as a safe
                # upper bound for `same_phase_next_beat` length checks.
                nf_list.append(int(meta.get("num_frames", fpc)))

            # Optional: in-batch HR derangement (shuffled-HR control).
            shuffled_info = None
            if self.shuffled_hr:
                shuffled_info = apply_shuffled_hr(hr_list, rng=rng)
                hr_list = shuffled_info.shuffled
                if shuffled_info.was_applied:
                    self.stats.add_shuffled_hr_applied(batch_size)

            collated_batch = torch.utils.data.default_collate(fpc_batch)
            collated_masks_enc_all: List[torch.Tensor] = []
            collated_masks_pred_all: List[torch.Tensor] = []

            for mgi, mask_generator in enumerate(self.mask_generators[fpc]):
                enc_list, pred_list = self._generate_phase_aware_masks(
                    mask_generator=mask_generator,
                    batch_size=batch_size,
                    hr_list=hr_list,
                    ft_list=ft_list,
                    nf_list=nf_list,
                    fpc=fpc,
                    HW=HW,
                    rng=rng,
                )
                collated_masks_enc_all.append(enc_list)
                collated_masks_pred_all.append(pred_list)

            fpc_collations += [
                (collated_batch, collated_masks_enc_all, collated_masks_pred_all)
            ]

        return fpc_collations

    def _generate_phase_aware_masks(
        self,
        mask_generator: "_MaskGenerator",
        batch_size: int,
        hr_list: List[float],
        ft_list: List[float],
        nf_list: List[int],
        fpc: int,
        HW: int,
        rng: np.random.Generator,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """For each sample: run the existing context sampler (vanilla), then
        re-sample the target block positions so the block's sampled-sequence
        Δφ lies in the chosen bucket's interval. Falls back to the vanilla
        target block on invalid metadata or rejection failure.
        """
        # Block size drawn once per mask_generator call, matching vanilla.
        g_size = torch.Generator()
        g_size.manual_seed(mask_generator.step())
        p_size = mask_generator._sample_block_size(
            generator=g_size,
            temporal_scale=mask_generator.temporal_pred_mask_scale,
            spatial_scale=mask_generator.spatial_pred_mask_scale,
            aspect_ratio_scale=mask_generator.aspect_ratio,
        )
        t_blk, h_blk, w_blk = p_size
        npred = mask_generator.npred
        D = mask_generator.duration
        H = mask_generator.height
        W = mask_generator.width

        masks_enc_batch: List[torch.Tensor] = []
        masks_pred_batch: List[torch.Tensor] = []
        min_keep_enc = min_keep_pred = D * H * W

        for bi in range(batch_size):
            hr = hr_list[bi]
            ft = ft_list[bi]
            nf = nf_list[bi]
            ok, _reason = validate_hr(hr, ft, nf) if self.require_valid_hr else (True, "ok")
            self.stats.add_clips(1)
            if ok:
                self.stats.add_valid_meta(1)
                cycle_tubelets = cycle_tubelets_from_hr(
                    hr_bpm=hr,
                    fps_sampled=self.fps_sampled,
                    tubelet_size=mask_generator.temporal_patch_size,
                )
                self.stats.push_cycle([float(cycle_tubelets)])
            else:
                cycle_tubelets = None

            # 1) Context via the vanilla sampler (intersection of npred blocks).
            empty_context = True
            attempt = 0
            ctx_mask_binary = None
            chosen_ctx_block_coords = None
            while empty_context and attempt < 50:
                attempt += 1
                mask_e_bin = torch.ones((D, H, W), dtype=torch.int32)
                # Track the individual block coords (to use as reference
                # "context block" for phase math).
                block_list = []
                for _ in range(npred):
                    top = int(torch.randint(0, H - h_blk + 1, (1,)).item())
                    left = int(torch.randint(0, W - w_blk + 1, (1,)).item())
                    t_start = int(torch.randint(0, D - t_blk + 1, (1,)).item())
                    block = torch.ones((D, H, W), dtype=torch.int32)
                    block[t_start:t_start + t_blk, top:top + h_blk, left:left + w_blk] = 0
                    mask_e_bin *= block
                    block_list.append((t_start, top, left))

                if mask_generator.max_context_duration < D:
                    mask_e_bin[mask_generator.max_context_duration:, :, :] = 0

                empty_context = int(mask_e_bin.sum().item()) == 0
                if not empty_context:
                    ctx_mask_binary = mask_e_bin
                    chosen_ctx_block_coords = block_list

            if ctx_mask_binary is None:
                # Vanilla path also gives up here: fall back one more time
                # with pure vanilla to preserve invariant.
                masks_enc_bi, masks_pred_bi = self._vanilla_sample_one(
                    mask_generator, p_size
                )
                masks_enc_batch.append(masks_enc_bi)
                masks_pred_batch.append(masks_pred_bi)
                min_keep_enc = min(min_keep_enc, masks_enc_bi.numel())
                min_keep_pred = min(min_keep_pred, masks_pred_bi.numel())
                self.stats.add_fallback_invalid_meta(1)
                continue

            # Flatten vanilla context mask (tokens to KEEP) + predictor indices
            # (tokens to PREDICT) exactly like vanilla sampler.
            vanilla_mask_p = torch.argwhere(ctx_mask_binary.flatten() == 0).squeeze(-1)
            vanilla_mask_e = torch.nonzero(ctx_mask_binary.flatten()).squeeze(-1)

            # 2) If phase-aware fails or metadata invalid, fall back to vanilla.
            if not ok:
                masks_enc_batch.append(vanilla_mask_e)
                masks_pred_batch.append(vanilla_mask_p)
                min_keep_enc = min(min_keep_enc, vanilla_mask_e.numel())
                min_keep_pred = min(min_keep_pred, vanilla_mask_p.numel())
                self.stats.add_fallback_invalid_meta(1)
                continue

            # 3) Phase-aware target re-sampling.
            new_pred_flat = self._sample_phase_aware_targets(
                ctx_mask_binary=ctx_mask_binary,
                ctx_block_list=chosen_ctx_block_coords,
                p_size=p_size,
                cycle_tubelets=cycle_tubelets,
                D=D, H=H, W=W,
                rng=rng,
            )

            if new_pred_flat is None:
                # Phase-aware failed after max attempts on every bucket.
                masks_enc_batch.append(vanilla_mask_e)
                masks_pred_batch.append(vanilla_mask_p)
                min_keep_enc = min(min_keep_enc, vanilla_mask_e.numel())
                min_keep_pred = min(min_keep_pred, vanilla_mask_p.numel())
                self.stats.add_fallback_bucket_fail(1)
                continue

            # new_pred_flat overrides the predictor mask, but the encoder
            # (context) mask is the vanilla complement of the context blocks
            # picked above -- we do NOT alter the context side.
            masks_enc_batch.append(vanilla_mask_e)
            masks_pred_batch.append(new_pred_flat)
            min_keep_enc = min(min_keep_enc, vanilla_mask_e.numel())
            min_keep_pred = min(min_keep_pred, new_pred_flat.numel())

        # Apply max_keep like vanilla.
        if mask_generator.max_keep is not None:
            min_keep_enc = min(min_keep_enc, mask_generator.max_keep)
        masks_enc_batch = [cm[:min_keep_enc] for cm in masks_enc_batch]
        masks_pred_batch = [cm[:min_keep_pred] for cm in masks_pred_batch]

        # full_complement / pred_full_complement mirror vanilla logic.
        if mask_generator.full_complement:
            total = D * H * W
            masks_pred_batch = [
                torch.tensor(
                    sorted(list(set(range(total)) - set(cm.tolist()))),
                    dtype=cm.dtype,
                )
                for cm in masks_enc_batch
            ]
        elif mask_generator.pred_full_complement:
            total = D * H * W
            masks_enc_batch = [
                torch.tensor(
                    sorted(list(set(range(total)) - set(cm.tolist()))),
                    dtype=cm.dtype,
                )
                for cm in masks_pred_batch
            ]

        masks_enc_t = torch.utils.data.default_collate(masks_enc_batch)
        masks_pred_t = torch.utils.data.default_collate(masks_pred_batch)

        if mask_generator.inv_block:
            return masks_pred_t, masks_enc_t
        else:
            return masks_enc_t, masks_pred_t

    def _vanilla_sample_one(
        self,
        mask_generator: "_MaskGenerator",
        p_size: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-sample vanilla mask. Used as a last-resort fallback."""
        empty = True
        while empty:
            mask = torch.ones(
                (mask_generator.duration, mask_generator.height, mask_generator.width),
                dtype=torch.int32,
            )
            for _ in range(mask_generator.npred):
                mask *= mask_generator._sample_block_mask(p_size)
            flat = mask.flatten()
            mp = torch.argwhere(flat == 0).squeeze(-1)
            me = torch.nonzero(flat).squeeze(-1)
            empty = me.numel() == 0
            if not empty:
                return me, mp

    def _sample_phase_aware_targets(
        self,
        ctx_mask_binary: torch.Tensor,
        ctx_block_list: List[Tuple[int, int, int]],
        p_size: Tuple[int, int, int],
        cycle_tubelets: float,
        D: int,
        H: int,
        W: int,
        rng: np.random.Generator,
    ) -> Optional[torch.Tensor]:
        """Sample a new set of target blocks whose circular-mean phase offset
        from the context reference falls within the chosen bucket. Context
        reference phase is the circular mean of block-center phases over the
        context blocks (preserves whatever the context sampler produced).

        Returns a flat 1-D tensor of predictor-token indices, or None if
        rejection sampling exhausts attempts on every bucket.
        """
        t_blk, h_blk, w_blk = p_size

        # Context reference phi: mean of context block center phases.
        ctx_phis = [
            block_center_phi(t, t_blk, cycle_tubelets)
            for (t, _top, _left) in ctx_block_list
        ]
        # Simple arithmetic circular combine of ctx block phis.
        from src.masks.phase_mask_utils import circular_mean as _cm
        phi_c = _cm(ctx_phis)
        ctx_center_time = float(np.mean([
            block_center_time_tubelets(t, t_blk) for (t, _, _) in ctx_block_list
        ]))

        # Build list of candidate t_starts for a target block of temporal
        # length t_blk (on the sampled grid).
        valid_starts = np.arange(0, max(1, D - t_blk + 1), dtype=np.int64)
        if valid_starts.size == 0:
            return None

        # Precompute per-start phi (block center) and time-distance from
        # ctx_center_time.
        phis_at_start = np.asarray(
            [block_center_phi(int(s), t_blk, cycle_tubelets) for s in valid_starts],
            dtype=np.float64,
        )
        t_centers = np.asarray(
            [block_center_time_tubelets(int(s), t_blk) for s in valid_starts],
            dtype=np.float64,
        )

        # Fallback retry order: only include buckets with positive probability
        # so that zero-prob buckets (e.g. user disabling a bucket for an
        # ablation) are never silently used.
        enabled_names = [n for n, p in self.phase_probs.items() if p > 0.0]
        if not enabled_names:
            return None
        first = choose_bucket(self.phase_probs, rng)
        order = [first] + [n for n in enabled_names if n != first]
        tried_buckets: set = set()

        for bucket_name in order:
            if bucket_name in tried_buckets:
                continue
            tried_buckets.add(bucket_name)
            spec = self.phase_specs[bucket_name]

            # Build the set of valid starts for this bucket.
            if spec.next_beat:
                # target center must be ~1 cycle away (in tubelets) from the
                # context center, and its phi must be within tolerance of phi_c.
                min_time_gap = max(1.0, cycle_tubelets - spec.next_beat_tolerance * cycle_tubelets)
                max_time_gap = cycle_tubelets + spec.next_beat_tolerance * cycle_tubelets
                time_gap = np.abs(t_centers - ctx_center_time)
                time_ok = (time_gap >= min_time_gap) & (time_gap <= max_time_gap)
                # Phase near ctx phi (small circular distance).
                phase_dist = np.asarray(
                    [circular_dist(float(p), float(phi_c)) for p in phis_at_start],
                    dtype=np.float64,
                )
                phase_ok = phase_dist <= spec.next_beat_tolerance
                candidate_mask = time_ok & phase_ok
                if not candidate_mask.any():
                    self.stats.add_same_phase_skipped(1)
                    self.stats.inc_bucket_fail(bucket_name)
                    continue
            else:
                dphis = np.asarray(
                    [dphi_fraction(float(phi_c), float(p)) for p in phis_at_start],
                    dtype=np.float64,
                )
                candidate_mask = (dphis >= spec.lo) & (dphis <= spec.hi)
                if not candidate_mask.any():
                    self.stats.inc_bucket_fail(bucket_name)
                    continue

            cand_starts = valid_starts[candidate_mask]

            # Now pick npred target blocks whose spatial positions are unseen;
            # we keep spatial sampling random (only temporal is phase-gated).
            new_pred_bin = torch.zeros((D, H, W), dtype=torch.int32)
            npred = len(ctx_block_list)
            dphi_for_stats: List[float] = []
            succeeded = True
            for _ in range(npred):
                t_start = int(rng.choice(cand_starts))
                top = int(rng.integers(0, H - h_blk + 1))
                left = int(rng.integers(0, W - w_blk + 1))
                new_pred_bin[t_start:t_start + t_blk, top:top + h_blk, left:left + w_blk] = 1
                # Record dphi for the chosen start (for logging).
                if spec.next_beat:
                    dphi_for_stats.append(float(abs(circular_dist(
                        block_center_phi(t_start, t_blk, cycle_tubelets), phi_c
                    ))))
                else:
                    dphi_for_stats.append(float(dphi_fraction(
                        phi_c, block_center_phi(t_start, t_blk, cycle_tubelets)
                    )))

            if not succeeded:
                self.stats.inc_bucket_fail(bucket_name)
                continue

            # Respect max_context_duration for predictor targets too.
            new_pred_flat = torch.argwhere(new_pred_bin.flatten() == 1).squeeze(-1)
            if new_pred_flat.numel() == 0:
                self.stats.inc_bucket_fail(bucket_name)
                continue

            self.stats.inc_bucket(bucket_name)
            self.stats.push_dphi(dphi_for_stats)
            return new_pred_flat

        return None

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def __call__(self, batch):
        if self.phase_aware:
            return self._phase_call(batch)
        return self._vanilla_call(batch)


class _MaskGenerator(object):

    def __init__(
        self,
        crop_size=(224, 224),
        num_frames=16,
        spatial_patch_size=(16, 16),
        temporal_patch_size=2,
        spatial_pred_mask_scale=(0.2, 0.8),
        temporal_pred_mask_scale=(1.0, 1.0),
        aspect_ratio=(0.3, 3.0),
        npred=1,
        max_context_frames_ratio=1.0,
        max_keep=None,
        inv_block=False,
        full_complement=False,
        pred_full_complement=False,
    ):
        super(_MaskGenerator, self).__init__()
        if not isinstance(crop_size, tuple):
            crop_size = (crop_size,) * 2
        if not isinstance(spatial_patch_size, tuple):
            spatial_patch_size = (spatial_patch_size,) * 2
        self.crop_size = crop_size
        self.height, self.width = [crop_size[i] // spatial_patch_size[i] for i in (0, 1)]
        self.duration = num_frames // temporal_patch_size
        self.full_complement = full_complement
        self.pred_full_complement = pred_full_complement

        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.aspect_ratio = aspect_ratio
        self.spatial_pred_mask_scale = spatial_pred_mask_scale
        self.temporal_pred_mask_scale = temporal_pred_mask_scale
        self.npred = npred
        self.max_context_duration = max(
            1, int(self.duration * max_context_frames_ratio)
        )  # maximum number of time-steps (frames) spanned by context mask
        self.max_keep = max_keep  # maximum number of patches to keep in context
        self._itr_counter = Value("i", -1)  # collator is shared across worker processes
        self.inv_block = inv_block

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, temporal_scale, spatial_scale, aspect_ratio_scale):
        # -- Sample temporal block mask scale
        _rand = torch.rand(1, generator=generator).item()
        min_t, max_t = temporal_scale
        temporal_mask_scale = min_t + _rand * (max_t - min_t)
        t = max(1, int(self.duration * temporal_mask_scale))

        # -- Sample spatial block mask scale
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = spatial_scale
        spatial_mask_scale = min_s + _rand * (max_s - min_s)
        spatial_num_keep = int(self.height * self.width * spatial_mask_scale)

        # -- Sample block aspect-ratio
        _rand = torch.rand(1, generator=generator).item()
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)

        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(spatial_num_keep * aspect_ratio)))
        w = int(round(math.sqrt(spatial_num_keep / aspect_ratio)))
        h = min(h, self.height)
        w = min(w, self.width)

        return (t, h, w)

    def _sample_block_mask(self, b_size):
        t, h, w = b_size
        top = torch.randint(0, self.height - h + 1, (1,))
        left = torch.randint(0, self.width - w + 1, (1,))
        start = torch.randint(0, self.duration - t + 1, (1,))

        mask = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
        mask[start : start + t, top : top + h, left : left + w] = 0

        # Context mask will only span the first X frames
        # (X=self.max_context_frames)
        if self.max_context_duration < self.duration:
            mask[self.max_context_duration :, :, :] = 0

        # --
        return mask

    def __call__(self, batch_size):
        """
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample pred block size using seed
        # 2. sample several pred block locations for each image (w/o seed)
        # 3. return pred masks and complement (enc mask)
        """
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            temporal_scale=self.temporal_pred_mask_scale,
            spatial_scale=self.spatial_pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio,
        )

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_enc = min_keep_pred = self.duration * self.height * self.width
        for _ in range(batch_size):

            empty_context = True
            while empty_context:

                mask_e = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
                for _ in range(self.npred):
                    mask_e *= self._sample_block_mask(p_size)
                mask_e = mask_e.flatten()

                mask_p = torch.argwhere(mask_e == 0).squeeze()
                mask_e = torch.nonzero(mask_e).squeeze()

                empty_context = len(mask_e) == 0
                if not empty_context:
                    min_keep_pred = min(min_keep_pred, len(mask_p))
                    min_keep_enc = min(min_keep_enc, len(mask_e))
                    collated_masks_pred.append(mask_p)
                    collated_masks_enc.append(mask_e)

        if self.max_keep is not None:
            min_keep_enc = min(min_keep_enc, self.max_keep)

        collated_masks_enc = [cm[:min_keep_enc] for cm in collated_masks_enc]
        collated_masks_pred = [cm[:min_keep_pred] for cm in collated_masks_pred]
        if self.full_complement:  # predictor mask is just complement of encoder mask
            collated_masks_pred = [
                torch.tensor(
                    sorted(list(set(range(int(self.duration * self.height * self.width))) - set(cm.tolist()))),
                    dtype=cm.dtype,
                )
                for cm in collated_masks_enc
            ]
        elif self.pred_full_complement:
            collated_masks_enc = [
                torch.tensor(
                    sorted(list(set(range(int(self.duration * self.height * self.width))) - set(cm.tolist()))),
                    dtype=cm.dtype,
                )
                for cm in collated_masks_pred
            ]

        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)

        if self.inv_block:
            return collated_masks_pred, collated_masks_enc  # predict context from block
        else:
            return collated_masks_enc, collated_masks_pred
