"""Generic EMA helper.

Single implementation used by both ``StudyTransformerEMA`` (EchoMV-JEPA) and
re-exported from ``src.models.study_projectors`` via the existing
``EMAProjectorPair.update_teacher`` method (which inlines the same math). We
use ``torch._foreach_mul_/add_`` for speed, matching ``app/vjepa/train.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def ema_update_(teacher: nn.Module, student: nn.Module, tau: float) -> None:
    """In-place EMA update: ``teacher <- tau * teacher + (1 - tau) * student``.

    Parameter shape / ordering must match one-to-one, which is guaranteed when
    teacher is ``copy.deepcopy(student)`` and neither has been surgically
    modified since.
    """
    params_s = [p for p in student.parameters()]
    params_t = [p for p in teacher.parameters()]
    if len(params_s) != len(params_t):
        raise RuntimeError(f"ema_update_: param count mismatch teacher={len(params_t)} student={len(params_s)}")
    torch._foreach_mul_(params_t, tau)
    torch._foreach_add_(params_t, params_s, alpha=1.0 - tau)


__all__ = ["ema_update_"]
