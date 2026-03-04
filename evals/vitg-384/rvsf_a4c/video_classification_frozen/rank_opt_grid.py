#!/usr/bin/env python3
"""
Rank opt_grid configs by per-head accuracies stored in a checkpoint, using an experiment directory.

Adds:
  --print-grid-dicts   -> pretty-print the opt_grid dict for each top-K grid
  --print-all-grid-dicts -> pretty-print opt_grid dicts for all ranked grids

Examples:
  python3 rank_opt_grid.py --exp-dir EXP --epoch 2 --metric best_val_acc_per_head --topk 10 --print-grid-dicts
  python3 rank_opt_grid.py --exp-dir EXP --epoch 2 --metric best_val_acc_per_head --print-all-grid-dicts
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any
from pprint import pprint


def grid_idx_for_head(h: int, H: int, G: int, mapping: str) -> int:
    if mapping == "direct":
        return min(h, G - 1)
    if mapping == "mod":
        return h % G
    if mapping == "auto":
        if H == G:
            return h
        if H % G == 0:
            return h % G
        return min(h, G - 1)
    raise ValueError(f"Unknown mapping: {mapping}")


def ckpt_path_from_exp(exp_dir: Path, epoch: int | None, ckpt_name: str) -> Path:
    if ckpt_name:
        return exp_dir / ckpt_name
    if epoch is None:
        p = exp_dir / "epoch_002.pt"
        if p.exists():
            return p
        p = exp_dir / "latest.pt"
        if p.exists():
            return p
        raise FileNotFoundError(f"No epoch_002.pt or latest.pt found under {exp_dir}")
    return exp_dir / f"epoch_{epoch:03d}.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="", help="Direct path to a checkpoint .pt (overrides --exp-dir/--epoch/--ckpt-name)")
    ap.add_argument("--exp-dir", default="", help="Experiment directory containing checkpoints (epoch_XXX.pt)")
    ap.add_argument("--epoch", type=int, default=None, help="Epoch number to load (e.g., 2 -> epoch_002.pt)")
    ap.add_argument("--ckpt-name", default="", help="Override checkpoint filename (e.g., latest.pt)")
    ap.add_argument(
        "--metric",
        default="best_val_acc_per_head",
        help="Key in checkpoint for per-head accuracy (e.g. best_val_acc_per_head or val_acc_per_head)",
    )
    ap.add_argument("--grid-key", default="opt_grid", help="Key in checkpoint for the grid list")
    ap.add_argument(
        "--mapping",
        default="auto",
        choices=["auto", "mod", "direct"],
        help="How to map head->grid when #heads != #grids. auto: if H%G==0 use mod else clamp.",
    )
    ap.add_argument("--topk", type=int, default=10, help="How many grids to print")
    ap.add_argument(
        "--rank-by",
        default="max",
        choices=["max", "mean"],
        help="Primary ranking criterion per-grid",
    )
    ap.add_argument(
        "--secondary",
        default="mean",
        choices=["none", "max", "mean"],
        help="Secondary tie-breaker",
    )
    ap.add_argument("--out", default=None, help="Optional CSV output path (relative to exp-dir if not absolute)")
    ap.add_argument(
        "--device",
        default="meta",
        choices=["meta", "cpu"],
        help="meta avoids loading tensors; cpu loads fully (can be huge).",
    )
    ap.add_argument(
        "--print-grid-dicts",
        action="store_true",
        help="Pretty-print the opt_grid dict(s) for the top-K ranked grids.",
    )
    ap.add_argument(
        "--print-all-grid-dicts",
        action="store_true",
        help="Pretty-print the opt_grid dict(s) for ALL ranked grids (can be long).",
    )
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir).expanduser().resolve()
    if not exp_dir.exists():
        raise SystemExit(f"exp-dir does not exist: {exp_dir}")

    if args.ckpt:
        ckpt_path = Path(args.ckpt).expanduser().resolve()
        exp_dir = ckpt_path.parent
    else:
        if not args.exp_dir:
            raise SystemExit("Provide either --ckpt or --exp-dir")
        exp_dir = Path(args.exp_dir).expanduser().resolve()
        if not exp_dir.exists():
            raise SystemExit(f"exp-dir does not exist: {exp_dir}")
        ckpt_path = ckpt_path_from_exp(exp_dir, args.epoch, args.ckpt_name)
    
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")


    import torch

    ckpt = torch.load(str(ckpt_path), map_location=args.device)

    if args.metric not in ckpt:
        raise SystemExit(f"Metric key '{args.metric}' not found. Available keys: {sorted(ckpt.keys())}")
    if args.grid_key not in ckpt:
        raise SystemExit(f"Grid key '{args.grid_key}' not found. Available keys: {sorted(ckpt.keys())}")

    acc = list(ckpt[args.metric])
    grid = list(ckpt[args.grid_key])

    H, G = len(acc), len(grid)
    if G == 0:
        raise SystemExit("opt_grid is empty.")

    per: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for h, a in enumerate(acc):
        gi = grid_idx_for_head(h, H, G, args.mapping)
        per[gi].append((h, float(a)))

    rows: list[dict[str, Any]] = []
    for gi in range(G):
        heads = per.get(gi, [])
        if not heads:
            continue
        best_h, best_a = max(heads, key=lambda x: x[1])
        mean_a = sum(a for _, a in heads) / len(heads)
        g = grid[gi]

        rows.append(
            {
                "grid_idx": gi,
                "n_heads": len(heads),
                "max_acc": best_a,
                "max_head": best_h,
                "mean_acc": mean_a,
                "ref_lr": g.get("ref_lr"),
                "ref_wd": g.get("ref_wd"),
                "start_lr": g.get("start_lr"),
                "final_lr": g.get("final_lr"),
                "final_wd": g.get("final_wd"),
                "warmup": g.get("warmup"),
            }
        )

    def keyfn(r):
        primary = r["max_acc"] if args.rank_by == "max" else r["mean_acc"]
        if args.secondary == "none":
            secondary = 0.0
        elif args.secondary == "max":
            secondary = r["max_acc"]
        else:
            secondary = r["mean_acc"]
        return (primary, secondary)

    rows.sort(key=keyfn, reverse=True)

    topk = min(args.topk, len(rows))
    print(f"exp_dir={exp_dir}")
    print(f"ckpt={ckpt_path}")
    print(f"metric={args.metric}  grid_key={args.grid_key}  device={args.device}")
    print(f"Heads={H}  Grids={G}  mapping={args.mapping}")
    print("")
    print(f"Top {topk} grids (rank_by={args.rank_by}, secondary={args.secondary}):")
    for i, r in enumerate(rows[:topk], 1):
        print(
            f"{i:2d}. grid={r['grid_idx']:2d} "
            f"max={r['max_acc']:.4f} (head {r['max_head']}) "
            f"mean={r['mean_acc']:.4f} over {r['n_heads']} heads "
            f"ref_lr={r['ref_lr']} ref_wd={r['ref_wd']} warmup={r['warmup']}"
        )

    # Pretty-print grid dicts in the same format as pprint(opt_grid)
    if args.print_all_grid_dicts or args.print_grid_dicts:
        to_print = rows if args.print_all_grid_dicts else rows[:topk]
        print("\nGrid dicts:")
        for rank, r in enumerate(to_print, 1):
            gi = r["grid_idx"]
            print(f"\n# {rank}. grid={gi} max_acc={r['max_acc']:.4f} mean_acc={r['mean_acc']:.4f}")
            pprint(grid[gi], sort_dicts=False)

    if args.out:
        out_path = Path(args.out).expanduser()
        if not out_path.is_absolute():
            out_path = exp_dir / out_path

        fieldnames = [
            "grid_idx",
            "n_heads",
            "max_acc",
            "max_head",
            "mean_acc",
            "ref_lr",
            "ref_wd",
            "start_lr",
            "final_lr",
            "final_wd",
            "warmup",
        ]
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in fieldnames})
        print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
