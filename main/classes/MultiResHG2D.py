import torch
import torch.nn.functional as F
from torch import nn

from HashGrid2D import HashGrid2D

#Layout format: [(hash_size, cell_size, dimensions)]

class MRHG2D(nn.Module):

    def __init__(self, layout=None):
        super().__init__()
        if layout is None:
            layout = [(14, 16, 2)]
        self.layout = layout
        self.grids = nn.ModuleList([
            HashGrid2D(hash_size=size, cell_size=cell, dimensions=dim)
            for size, cell, dim in layout
        ])
        self.dimensions = sum(dim for _, _, dim in layout)
        self.level_weights = nn.Parameter(0.9 * torch.rand(len(layout)) + 0.1)

    def get_dimensions(self):
        return self.dimensions

    def forward(self, positions):
        features = []

        for i, (sdhg, (_, _, dim)) in enumerate(zip(self.grids, self.layout)):
            feat = sdhg.batch_lookup(positions)
            norm_feat = F.layer_norm(feat, (dim,))
            scaled_feat = self.level_weights[i] * norm_feat
            features.append(scaled_feat)

        return torch.cat(features, dim=1)
