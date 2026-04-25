# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn


class MultiSeqWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, masks=None):
        """
        :param x: [list] List of Tensors of different seq lengths
        :param masks: [list[list]] List of Tensors (out index: masks for given seq length, inner index: multimasks for that seq len)
        """
        if masks is None:
            return [self.backbone(xi) for xi in x]

        outs = [[] for _ in x]
        for i, (xi, mi) in enumerate(zip(x, masks)):
            for mij in mi:
                outs[i] += [self.backbone(xi, masks=mij)]
        return outs


class PredictorMultiSeqWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, masks_x, masks_y, has_cls=False, delta_phi=None):
        """
        :param delta_phi: Optional list matching masks_y structure (outer: fpc,
            inner: mask-generator). Each element is a [B, N_target] tensor of
            cycle-fraction offsets. When None, the predictor uses its standard
            (phase-unaware) path.
        """
        n = 0
        outs = [[] for _ in x]
        for i, (xi, mxi, myi) in enumerate(zip(x, masks_x, masks_y)):
            for j, (xij, mxij, myij) in enumerate(zip(xi, mxi, myi)):
                dphi_ij = None
                if delta_phi is not None:
                    dphi_ij = delta_phi[i][j]
                outs[i] += [self.backbone(xij, mxij, myij, mask_index=i, has_cls=has_cls, delta_phi=dphi_ij)]
                n += 1
        return outs
