"""Control D — Metadata-only study transformer (plan §7).

Transformer inputs are only meta tokens (view_family, modality, phase_bucket,
quality); zero ``c_clip`` content. Downstream probe trained on ``[STUDY]``.

Rules out: metadata priors explain downstream performance.

Implemented as a one-line config override (zeroing out the input element
vectors) plus the shared head. The actual training loop is the main Stage-2
loop; this file exposes the masking helper.
"""

from __future__ import annotations

import torch


def zero_element_content(ctx_elements: torch.Tensor) -> torch.Tensor:
    """Replace every context element's ``c_clip``-derived content with zeros.

    Meta tokens still provide ``(view, modality, phase, quality)`` signal at
    the element position; the element body carries no clip information.
    """
    return torch.zeros_like(ctx_elements)


__all__ = ["zero_element_content"]
