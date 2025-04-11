import math
import torch
import torch.nn as nn
from typing import Tuple


class HashGrid2D(nn.Module):

    def __init__(self, hash_size: int = 14, cell_size: float = 1, dimensions: int = 4):
        super().__init__()
        self.hash_size = 2 ** hash_size
        self.cell_size = cell_size
        self.dimensions = dimensions
        self.grid = nn.ParameterDict()

    def _bitmix_hash(self, ix: torch.Tensor, iy: torch.Tensor) -> torch.Tensor:
        """
        Vectorized bit-mixing hash for batched positions (Murmur-inspired).
        """
        h = ix
        h ^= (h >> 16)
        h *= 0x85ebca6b
        h ^= (h >> 13)
        h += iy * 0xc2b2ae35
        h ^= (h >> 16)
        return h % self.hash_size

    def _bitmix_hash_single(self, ix: int, iy: int) -> int:
        """
        Scalar version of the bit-mixing hash for single positions.
        """
        h = ix
        h ^= (h >> 16)
        h *= 0x85ebca6b
        h ^= (h >> 13)
        h += iy * 0xc2b2ae35
        h ^= (h >> 16)
        return h % self.hash_size

    def insert(self, position: Tuple[float, float], feature: torch.Tensor):
        index = self.quantize(position)
        self.grid[index] = nn.Parameter(feature)

    def lookup(self, position: Tuple[float, float]):
        index = self.quantize(position)
        if index not in self.grid:
            self.grid[index] = nn.Parameter(torch.randn(self.dimensions) * 1e-2)
        return self.grid[index]

    def forward(self, position):
        return self.lookup(position)

    def quantize(self, position: Tuple[float, float]) -> str:
        ix = int(math.floor(position[0] / self.cell_size))
        iy = int(math.floor(position[1] / self.cell_size))
        return str(self._bitmix_hash_single(ix, iy))

    def batch_lookup(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Batched lookup using consistent bit-mixing hash function.
        """
        device = positions.device
        ix = torch.floor(positions[:, 0] / self.cell_size).long()
        iy = torch.floor(positions[:, 1] / self.cell_size).long()

        hashed = self._bitmix_hash(ix, iy)
        keys = hashed.tolist()

        features = []
        for k in keys:
            key = str(k)
            if key not in self.grid:
                self.grid[key] = nn.Parameter(torch.randn(self.dimensions, device=device) * 1e-2)
            features.append(self.grid[key])
        return torch.stack(features)
