from HashGrid2D import HashGrid2D
import torch
from MLP import NeRFMLP

grid = HashGrid2D()
mlp = NeRFMLP(input_dim=2 + 4)

position = (1.3, 2.7)
position_tensor = torch.tensor(position)

feature = grid.lookup(position)

input_tensor = torch.cat([position_tensor, feature], dim=0)

output = mlp(input_tensor)

print("Output:", output)