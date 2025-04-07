from collections import defaultdict
from typing import Tuple
import math
import torch
import torch.nn as nn
import BeforeNeRF
import numpy as np

class HashGrid2D(nn.Module):

    def __init__(self, hash_size : int = 2**14, cell_size : float = 1, dimensions : int = 4):
        super().__init__()
        self.hash_size = hash_size
        self.cell_size = cell_size
        self.dimensions = dimensions
        self.grid = nn.ParameterDict()

    def _hash(self, position : Tuple[int, int]) -> str:
        prime1 = 73856093
        prime2 = 19349663
        return str((position[0] * prime1 ^ position[1] * prime2) % self.hash_size)  # 2D

    def insert(self, position : Tuple[float,float], feature : torch.Tensor):
        index = self.quantize(position)
        self.grid[index] = nn.Parameter(feature)

    def lookup(self, position: Tuple[float, float]):
        index = self.quantize(position)
        if index not in self.grid:
            self.grid[index] = nn.Parameter(torch.randn(self.dimensions) * 1e-2)
        return self.grid[index]

    def forward(self, position):
        return self.lookup(position)

    def quantize(self, position : Tuple[float, float]):
        ix = int(math.floor(position[0] / self.cell_size))
        iy = int(math.floor(position[1] / self.cell_size))

        return self._hash((ix, iy))
