"""Init-time contextualization probe for the EchoMV-JEPA teacher.

Does the EMA study transformer's ``forward_contextualized`` output at a target
slot actually depend on the other elements in the same study? Or is the
architecture routing input → output through the residual path so strongly that
the attention update is negligible, making contextualized ≈ isolated?

Run:
    python -m experiments.echomv_jepa.context_sensitivity_probe
    python -m experiments.echomv_jepa.context_sensitivity_probe --d_model 512 --n_layers 4 --n_heads 8
    python -m experiments.echomv_jepa.context_sensitivity_probe --trained_ckpt /path/to/teacher_st_state.pt

Reports, per study in the batch:
  - cosine(LN(z_full), LN(z_iso))        per target row; mean, min.
  - L2(z_full - z_iso) / L2(z_full)     per target row; mean.
  - per-layer ||attn_out|| / ||residual_in||  ratio (how much attention
    perturbs the stream vs what was already there).

If mean cosine > 0.98 and per-layer attention ratio < 0.1, the architecture
has a near-identity shortcut at init — the full-study pass is numerically
indistinguishable from the isolated pass, and the Stage-1 cosine regression
loss can be trivially satisfied without any contextualization.
"""

from __future__ import annotations

import argparse
import math
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from src.models.echomv_jepa import StudyTransformerEMA
from src.models.meta_embeddings import MODALITY_VOCAB, MetaDropout, MetaEmbeddings
from src.models.study_transformer import StudyTransformer, StudyTransformerConfig


def _synth_batch(
    B: int, M: int, d_clip: int, n_modalities: int, device: torch.device, seed: int
) -> Dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    el = torch.randn(B, M, d_clip, device=device)
    pad = torch.zeros(B, M, dtype=torch.bool, device=device)
    view = torch.randint(0, 6, (B, M), device=device)
    mod = torch.randint(0, max(n_modalities, 1), (B, M), device=device)
    phase = torch.randint(0, 3, (B, M), device=device)
    quality = torch.randint(0, 4, (B, M), device=device)
    return dict(el=el, pad=pad, view=view, modality=mod, phase=phase, quality=quality)


def _encode_teacher_meta(meta: MetaEmbeddings, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Teacher sees full truth metadata; `MetaEmbeddings.encode_context` applies
    # per-field dropout only in training mode. We run with meta.eval() to match
    # training_step_echomv's teacher path.
    prev = meta.training
    meta.eval()
    with torch.no_grad():
        out = meta.encode_context(batch["view"], batch["modality"], batch["phase"], batch["quality"])
    meta.train(prev)
    return out


def _layer_ratios(teacher: StudyTransformer, elements: torch.Tensor, meta_add: torch.Tensor) -> List[float]:
    """For each block, compute ||attn_update|| / ||input_stream|| on the first study.

    Manually mirrors StudyTransformer.forward_contextualized so we can intercept
    each block's residual magnitudes.
    """
    ratios: List[float] = []
    with torch.no_grad():
        B = elements.shape[0]
        x_elem = teacher.clip_in(elements) + meta_add
        x_study = teacher.study_token.expand(B, 1, -1)
        x = torch.cat([x_study, x_elem], dim=1)
        pad = torch.zeros(B, x.shape[1], dtype=torch.bool, device=x.device)
        for blk in teacher.blocks:
            y = blk.ln1(x)
            attn_out, _ = blk.attn(y, y, y, key_padding_mask=pad, need_weights=False)
            ratio = (attn_out.norm(dim=-1).mean() / (x.norm(dim=-1).mean() + 1e-8)).item()
            ratios.append(ratio)
            x = x + attn_out
            x = x + blk.ffn(blk.ln2(x))
    return ratios


def run_probe(
    *,
    d_clip: int = 1024,
    d_model: int = 512,
    n_layers: int = 4,
    n_heads: int = 8,
    B: int = 8,
    M: int = 6,
    n_modalities: int = 1,
    seed: int = 0,
    device: str = "cpu",
    trained_ckpt: str = "",
) -> Dict[str, Any]:
    dev = torch.device(device)

    cfg = StudyTransformerConfig(
        d_clip=d_clip,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_mult=4,
        dropout_ffn=0.0,
        dropout_attn=0.0,
        max_M=max(M, 8),
    )
    torch.manual_seed(seed)
    student = StudyTransformer(cfg).to(dev).eval()
    teacher_pair = StudyTransformerEMA(student).to(dev).eval()
    teacher = teacher_pair.teacher

    if trained_ckpt:
        sd = torch.load(trained_ckpt, map_location="cpu")
        if "teacher_study_transformer" in sd:
            sd = sd["teacher_study_transformer"]
        teacher.load_state_dict(sd, strict=True)

    meta = MetaEmbeddings(d_model=d_model, dropout=MetaDropout(0.0, 0.0, 0.0, 0.0)).to(dev).eval()

    batch = _synth_batch(B=B, M=M, d_clip=d_clip, n_modalities=n_modalities, device=dev, seed=seed)
    full_meta = _encode_teacher_meta(meta, batch)

    with torch.no_grad():
        z_full = teacher_pair.forward_contextualized(batch["el"], full_meta, batch["pad"])  # (B, M, d_model)
        z_iso = teacher_pair.forward_isolated(batch["el"], full_meta)                        # (B, M, d_model)

        # Compare every element (treat every position as a "target").
        z_full_flat = z_full.reshape(-1, d_model)
        z_iso_flat = z_iso.reshape(-1, d_model)
        z_full_ln = F.layer_norm(z_full_flat, z_full_flat.shape[-1:])
        z_iso_ln = F.layer_norm(z_iso_flat, z_iso_flat.shape[-1:])
        cos = F.cosine_similarity(z_full_ln, z_iso_ln, dim=-1)  # (B*M,)

        l2_diff = (z_full_flat - z_iso_flat).norm(dim=-1)
        l2_full = z_full_flat.norm(dim=-1)
        rel_l2 = (l2_diff / (l2_full + 1e-8))

    ratios = _layer_ratios(teacher, batch["el"], full_meta)

    return {
        "cos_mean": cos.mean().item(),
        "cos_min": cos.min().item(),
        "cos_max": cos.max().item(),
        "rel_l2_mean": rel_l2.mean().item(),
        "rel_l2_max": rel_l2.max().item(),
        "per_layer_attn_over_input_norm": ratios,
        "B": B,
        "M": M,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "trained": bool(trained_ckpt),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_clip", type=int, default=1024)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--M", type=int, default=6)
    ap.add_argument("--n_modalities", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--trained_ckpt", type=str, default="")
    args = ap.parse_args()

    out = run_probe(
        d_clip=args.d_clip,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        B=args.B,
        M=args.M,
        n_modalities=args.n_modalities,
        seed=args.seed,
        device=args.device,
        trained_ckpt=args.trained_ckpt,
    )
    print(f"=== Context sensitivity probe ({'TRAINED' if out['trained'] else 'INIT'}) ===")
    print(f"  arch:  d_model={out['d_model']} n_layers={out['n_layers']} n_heads={out['n_heads']}")
    print(f"  batch: B={out['B']} M={out['M']}")
    print(f"  cos(z_full, z_iso):  mean={out['cos_mean']:.4f}  min={out['cos_min']:.4f}  max={out['cos_max']:.4f}")
    print(f"  rel_l2:              mean={out['rel_l2_mean']:.4f}  max={out['rel_l2_max']:.4f}")
    print(f"  per-layer ||attn_out|| / ||input_stream||:")
    for i, r in enumerate(out["per_layer_attn_over_input_norm"]):
        print(f"    layer {i}: {r:.4f}")
    verdict = (
        "NEAR-IDENTITY SHORTCUT"
        if out["cos_mean"] > 0.98
        else ("HEALTHY (significant contextualization)" if out["cos_mean"] < 0.9 else "BORDERLINE")
    )
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
