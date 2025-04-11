import math
import torch
import torch.nn as nn
from typing import Tuple


class HashGrid2D(nn.Module):
    def __init__(self, hash_size=14, cell_size=1.0, dimensions=4):
        super().__init__()
        self.hash_size = 2 ** hash_size
        self.cell_size = cell_size
        self.dimensions = dimensions
        self.grid = nn.Parameter(torch.randn(self.hash_size, self.dimensions) * 1e-2)

    def _hash(self, ix: torch.Tensor, iy: torch.Tensor) -> torch.Tensor:
        h = ix
        h ^= (h >> 16)
        h *= 0x85ebca6b
        h ^= (h >> 13)
        h += iy * 0xc2b2ae35
        h ^= (h >> 16)
        return h % self.hash_size

    def batch_lookup(self, positions: torch.Tensor):
        ix = torch.floor(positions[:, 0] / self.cell_size).long()
        iy = torch.floor(positions[:, 1] / self.cell_size).long()
        index = self._hash(ix, iy)
        return self.grid[index]

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.batch_lookup(positions)