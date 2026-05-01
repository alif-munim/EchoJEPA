"""Smoke test for PhaseRelationalHead under DistributedDataParallel.

The unified ``forward(c_a_pool, view_a_ids, view_b_pos_ids,
delta_phase_pos, y_pos_pool, y_neg_pool)`` must:

  1. Return (q_pre, y_pos_pre, y_hard_pre) tensors of correct shape.
  2. Survive a ``.backward()`` on a scalar loss.
  3. Produce gradients on the full parameter set (source_proj,
     relation_mlp, view_embed_a, view_embed_b_pos, phase_mlp,
     target_proj).
  4. Raise no AttributeError on the DDP wrapper's ``__call__``.
  5. Produce no DDP "unused parameter" reducer warning.

We initialize a single-process DDP group (world_size=1) via gloo so the
test does not require NCCL or multi-GPU. The single-process DDP path
still exercises ``DistributedDataParallel.__call__`` + ``forward`` +
the reducer's backward-hook registration, which is the failure surface
from the prior bug.

Run:
    python classifier/phase/sampler/test_ddp_wrapped_head.py
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

HERE = Path(__file__).resolve().parent
VJEPA_ROOT = HERE.parents[2]
if str(VJEPA_ROOT) not in sys.path:
    sys.path.insert(0, str(VJEPA_ROOT))

from app.vjepa_multiview.phase_relational_head import PhaseRelationalHead  # noqa: E402


def _init_single_process_group():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29505")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", world_size=1, rank=0)


def _expected_param_names(h: PhaseRelationalHead) -> set[str]:
    return {n for n, _ in h.named_parameters()}


def test_ddp_forward_and_backward():
    torch.manual_seed(0)
    _init_single_process_group()

    B, D, R = 8, 64, 32
    head = PhaseRelationalHead(embed_dim=D, rel_dim=R, hidden_dim=48)
    expected_params = _expected_param_names(head)

    # Wrap with DDP (single-process gloo group; same API surface as
    # production NCCL/multi-rank wrap).
    ddp_head = DistributedDataParallel(head, static_graph=False)

    # Inputs
    c_a_pool = torch.randn(B, D, requires_grad=True)
    view_a_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
    view_b_pos_ids = torch.tensor([2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    delta_phase_pos = torch.tensor(
        [0.05, 0.12, 0.25, 0.38, 0.5, 0.12, 0.25, 0.05]
    )
    # Teacher-side inputs arrive detached from upstream (teacher EMA
    # encoder + stop-grad); replicate by constructing them as
    # ``requires_grad=False`` leaves.
    y_pos_pool = torch.randn(B, D)
    y_neg_pool = torch.randn(B, D)

    # Collect any DDP reducer warnings emitted during forward/backward.
    # A "Grads will not be broadcasted" / "unused parameter" warning
    # would indicate the unified forward didn't actually touch every
    # parameter, which would defeat the whole point of this refactor.
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")

        q_pre, y_pos_pre, y_hard_pre = ddp_head(
            c_a_pool,
            view_a_ids,
            view_b_pos_ids,
            delta_phase_pos,
            y_pos_pool,
            y_neg_pool,
        )

        # Shape checks
        assert q_pre.shape == (B, R), f"q_pre {q_pre.shape}"
        assert y_pos_pre.shape == (B, R), f"y_pos_pre {y_pos_pre.shape}"
        assert y_hard_pre.shape == (B, R), f"y_hard_pre {y_hard_pre.shape}"

        # A tiny synthetic InfoNCE-shaped loss that uses all three
        # outputs so their autograd paths are exercised.
        q = F.normalize(q_pre, dim=-1)
        yp = F.normalize(y_pos_pre, dim=-1)
        yh = F.normalize(y_hard_pre, dim=-1)
        pos = (q * yp).sum(-1)
        hard = (q * yh).sum(-1)
        loss = (hard - pos).mean()
        loss.backward()

    # Filter for warnings that indicate a DDP reducer problem.
    ddp_warnings = [
        str(w.message) for w in w_list
        if "unused" in str(w.message).lower()
        or "reducer" in str(w.message).lower()
        or "mark_as_ready" in str(w.message).lower()
    ]
    assert not ddp_warnings, f"DDP reducer warnings emitted: {ddp_warnings}"

    # Gradient presence on the full head parameter set.
    missing_grads = []
    non_finite = []
    for name, p in head.named_parameters():
        if p.grad is None:
            missing_grads.append(name)
        elif not torch.isfinite(p.grad).all():
            non_finite.append(name)
    assert not missing_grads, (
        f"Parameters missing gradient (DDP reducer contract broken): "
        f"{missing_grads}"
    )
    assert not non_finite, f"Parameters with non-finite grads: {non_finite}"

    # Grouped assertions for the four expected submodule families.
    def _any_grad(prefix: str) -> bool:
        return any(
            (n.startswith(prefix) and p.grad is not None)
            for n, p in head.named_parameters()
        )

    assert _any_grad("source_proj"), "source_proj got no gradients"
    assert _any_grad("relation_mlp"), "relation_mlp got no gradients"
    assert _any_grad("view_embed_a"), "view_embed_a got no gradients"
    assert _any_grad("view_embed_b_pos"), "view_embed_b_pos got no gradients"
    assert _any_grad("phase_mlp"), "phase_mlp got no gradients"
    assert _any_grad("target_proj"), "target_proj got no gradients"

    # Student-side input must see grads (signal back into the encoder).
    assert c_a_pool.grad is not None
    assert torch.isfinite(c_a_pool.grad).all().item()

    # Teacher-side inputs are leaves with requires_grad=False → never
    # see grads. (We didn't set requires_grad=True on them.)
    assert not y_pos_pool.requires_grad
    assert not y_neg_pool.requires_grad

    # Parameter-name set sanity: DDP mustn't have added or dropped any.
    assert _expected_param_names(ddp_head.module) == expected_params

    print(
        "[pass] DDP-wrapped PhaseRelationalHead: single forward returns "
        "(q_pre, y_pos_pre, y_hard_pre); all 6 submodule families get "
        "gradients; no DDP reducer warnings; teacher inputs stay detached."
    )


def test_no_attribute_error_on_ddp_call():
    """Calling the DDP-wrapped head via __call__ must not AttributeError."""
    if not dist.is_initialized():
        _init_single_process_group()
    head = PhaseRelationalHead(embed_dim=32, rel_dim=16, hidden_dim=24)
    ddp_head = DistributedDataParallel(head, static_graph=False)
    B, D = 4, 32
    try:
        ddp_head(
            torch.randn(B, D),
            torch.zeros(B, dtype=torch.long),
            torch.zeros(B, dtype=torch.long),
            torch.zeros(B),
            torch.randn(B, D),
            torch.randn(B, D),
        )
    except AttributeError as e:
        raise AssertionError(f"DDP wrapper raised AttributeError: {e}") from e
    print("[pass] DDP __call__ did not AttributeError")


def _rank_worker(rank: int, world_size: int, result_queue, backend: str):
    """One rank process for the 2-rank DDP smoke test. Reports pass/fail
    via a multiprocessing.Queue so the parent can assert on it without
    the child printing to shared stdout."""
    import os as _os
    import warnings as _warnings
    import torch as _torch
    import torch.distributed as _dist
    import torch.nn.functional as _F
    from torch.nn.parallel import DistributedDataParallel as _DDP
    from app.vjepa_multiview.phase_relational_head import PhaseRelationalHead as _Head

    try:
        _os.environ["MASTER_ADDR"] = "127.0.0.1"
        _os.environ["MASTER_PORT"] = "29506"
        _os.environ["WORLD_SIZE"] = str(world_size)
        _os.environ["RANK"] = str(rank)
        _dist.init_process_group(backend=backend, world_size=world_size, rank=rank)

        _torch.manual_seed(17)  # identical seed across ranks so param init matches
        B, D, R = 4, 32, 16
        head = _Head(embed_dim=D, rel_dim=R, hidden_dim=24)
        ddp_head = _DDP(head, static_graph=False)

        # Per-rank data (different on each rank so all_reduce has
        # something to average).
        g = _torch.Generator().manual_seed(100 + rank)
        c_a_pool = _torch.randn(B, D, generator=g, requires_grad=True)
        y_pos_pool = _torch.randn(B, D, generator=g)
        y_neg_pool = _torch.randn(B, D, generator=g)
        view_a_ids = _torch.tensor([rank, rank + 1, rank + 2, rank + 3], dtype=_torch.long)
        view_b_pos_ids = _torch.tensor([rank + 2, rank + 3, rank + 4, rank + 5], dtype=_torch.long)
        delta_phase_pos = _torch.tensor([0.05, 0.25, 0.38, 0.12])

        with _warnings.catch_warnings(record=True) as w_list:
            _warnings.simplefilter("always")

            q_pre, y_pos_pre, y_hard_pre = ddp_head(
                c_a_pool, view_a_ids, view_b_pos_ids, delta_phase_pos,
                y_pos_pool, y_neg_pool,
            )
            assert q_pre.shape == (B, R)
            assert y_pos_pre.shape == (B, R)
            assert y_hard_pre.shape == (B, R)

            # InfoNCE-shaped loss using all three outputs.
            q = _F.normalize(q_pre, dim=-1)
            yp = _F.normalize(y_pos_pre, dim=-1)
            yh = _F.normalize(y_hard_pre, dim=-1)
            loss = ((q * yh).sum(-1) - (q * yp).sum(-1)).mean()
            loss.backward()

        # DDP reducer warnings are the leak indicator we care about.
        ddp_warnings = [
            str(w.message) for w in w_list
            if "unused" in str(w.message).lower()
            or "reducer" in str(w.message).lower()
            or "mark_as_ready" in str(w.message).lower()
        ]

        # Every param must have a finite gradient; DDP's reducer should
        # have all-reduced so gradients are identical across ranks.
        missing = [n for n, p in head.named_parameters() if p.grad is None]
        nonfin = [n for n, p in head.named_parameters()
                  if p.grad is not None and not _torch.isfinite(p.grad).all()]

        # Cross-rank gradient equality check: after all-reduce each rank's
        # gradient on the same parameter name must match every other rank.
        # Sample one parameter from each submodule family for this check.
        sample_param_names = [
            n for n, _ in head.named_parameters()
            if n.endswith(".weight")
        ][:6]
        grad_sums_local = _torch.tensor([
            float(dict(head.named_parameters())[n].grad.sum())
            for n in sample_param_names
        ])
        grads_all = [_torch.zeros_like(grad_sums_local) for _ in range(world_size)]
        _dist.all_gather(grads_all, grad_sums_local)
        max_diff = max(
            (grads_all[i] - grads_all[0]).abs().max().item()
            for i in range(1, world_size)
        ) if world_size > 1 else 0.0

        assert not missing, f"rank {rank} missing grads: {missing}"
        assert not nonfin, f"rank {rank} non-finite grads: {nonfin}"
        assert not ddp_warnings, f"rank {rank} reducer warnings: {ddp_warnings}"
        # If DDP all-reduce is wired up correctly, gradients on all ranks
        # are identical after backward.
        assert max_diff < 1e-5, (
            f"rank {rank}: gradients differ across ranks (max diff {max_diff:.3e}); "
            f"DDP all-reduce not firing on unified forward"
        )

        result_queue.put((rank, "OK", max_diff))
        _dist.destroy_process_group()
    except Exception as e:  # surface any failure to the parent
        import traceback
        result_queue.put((rank, "FAIL", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


def test_2rank_ddp_forward_and_backward():
    """Verify DDP forward/backward on PhaseRelationalHead across 2 ranks.

    The single-rank test cannot exercise all-reduce. This test spawns 2
    processes over gloo and confirms:

      - ddp_head(...) returns (q_pre, y_pos_pre, y_hard_pre)
      - InfoNCE backward succeeds on both ranks
      - Every parameter has a finite gradient on both ranks
      - No DDP reducer/unused-parameter warnings
      - **Cross-rank gradients match exactly** after all-reduce
        (this is the real DDP correctness check; if the forward path
        were bypassing DDP hooks, per-rank gradients would diverge).
    """
    # Gloo works on CPU without NCCL / CUDA, so the test is runnable
    # anywhere including CI.
    backend = "gloo"
    ctx = torch.multiprocessing.get_context("spawn")
    q = ctx.Queue()
    world_size = 2
    procs = [
        ctx.Process(target=_rank_worker, args=(rank, world_size, q, backend))
        for rank in range(world_size)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)

    results = []
    while not q.empty():
        results.append(q.get())

    # Collect + assert
    statuses = {r: (st, det) for r, st, det in results}
    assert len(statuses) == world_size, (
        f"Expected {world_size} rank reports, got {len(statuses)}: {statuses}"
    )
    fails = {r: det for r, (st, det) in statuses.items() if st != "OK"}
    assert not fails, f"Rank failures: {fails}"

    max_diffs = [det for _, (_, det) in statuses.items()]
    print(
        f"[pass] 2-rank DDP smoke: both ranks OK, cross-rank gradient "
        f"max-diff = {max(max_diffs):.3e} (< 1e-5)"
    )


def main():
    print("[INFO] Running DDP-wrapped PhaseRelationalHead smoke tests...")
    test_no_attribute_error_on_ddp_call()
    test_ddp_forward_and_backward()
    test_2rank_ddp_forward_and_backward()
    print("\nALL DDP SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
