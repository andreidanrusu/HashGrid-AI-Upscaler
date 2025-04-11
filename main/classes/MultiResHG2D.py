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
        self.level_weights = nn.Parameter(torch.rand(len(layout)))

    def get_dimensions(self):
        return self.dimensions

    def forward(self, positions):
        norm_weights = F.softmax(self.level_weights, dim=0)

        features = []

        for i, (sdhg, (_, _, dim)) in enumerate(zip(self.grids, self.layout)):
            norm_feat = sdhg.forward(positions)
            scaled_feat = norm_weights[i] * norm_feat
            features.append(scaled_feat)

        return torch.cat(features, dim=1)
