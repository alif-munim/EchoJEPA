"""Adapt MCC / full-joint training checkpoints for the Nature Medicine
probe eval pipeline.

The probe eval (`evals/video_classification_frozen/eval.py:load_pretrained`)
expects:
  * a dict with an ``encoder`` OR ``target_encoder`` key holding the ViT-L
    state_dict (module./backbone. prefixes are stripped automatically)
  * a top-level ``epoch`` key (only used for logging)

MCC checkpoints already have ``target_encoder`` + ``epoch`` → pass-through.
Full-joint checkpoints have ``clip_target_encoder`` + ``step`` → need to
re-save as ``target_encoder`` + ``epoch``.

Writes the adapted checkpoint as ``<stem>_for_eval.pt`` next to the source.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def adapt(src: Path, use_teacher: bool = True) -> Path:
    sd = torch.load(src, map_location="cpu", weights_only=False)
    top_keys = set(sd.keys())

    # Case 1: MCC-style (already has target_encoder). Re-save as the _for_eval
    # sibling anyway — downstream launcher expects the canonical _for_eval.pt
    # path so it can be overwritten between intermediate and final checkpoints.
    if "target_encoder" in top_keys and "epoch" in top_keys:
        dst = src.with_name(src.stem + "_for_eval.pt")
        # Simplest path: copy via torch save of the same dict (small effort; the
        # dict-level serialization is fast even for 5 GB because it's just
        # tensor references). This ensures the launcher's uniform output path.
        adapted = {
            "encoder": sd.get("encoder", sd["target_encoder"]),
            "target_encoder": sd["target_encoder"],
            "epoch": int(sd.get("epoch", 0)),
        }
        torch.save(adapted, dst)
        print(f"[{src.name}] MCC-style (already eval-compat); copied to {dst.name}")
        return dst

    # Case 2: Full-joint style (clip_target_encoder + step).
    if "clip_target_encoder" in top_keys:
        source_key = "clip_target_encoder" if use_teacher else "clip_encoder"
        print(f"[{src.name}] full-joint → eval adapter (using {source_key})")
        adapted = {
            "target_encoder": sd[source_key],
            "encoder": sd.get("clip_encoder", sd[source_key]),  # fallback same
            "epoch": int(sd.get("step", 0)),
            "step": int(sd.get("step", 0)),
        }
        dst = src.with_name(src.stem + "_for_eval.pt")
        torch.save(adapted, dst)
        print(f"  wrote {dst} (encoder keys: {len(adapted['target_encoder'])})")
        return dst

    raise ValueError(f"Don't know how to adapt {src} with top keys: {sorted(top_keys)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--use-online", action="store_true", help="Use online encoder (not teacher) as target_encoder")
    args = ap.parse_args()
    adapt(args.checkpoint, use_teacher=not args.use_online)


if __name__ == "__main__":
    main()
