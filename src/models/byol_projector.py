import torch.nn as nn


class BYOLProjector(nn.Module):
    """MLP projector: Linear -> BN -> ReLU -> Linear.

    Used on both online and target branches of BYOL.
    """

    def __init__(self, embed_dim=1024, hidden_dim=4096, proj_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x):
        return self.net(x)


class BYOLPredictor(nn.Module):
    """MLP predictor: Linear -> BN -> ReLU -> Linear.

    Only used on the online branch. Prevents representation collapse.
    """

    def __init__(self, proj_dim=256, hidden_dim=4096, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
