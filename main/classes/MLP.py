import torch.nn as nn

class NeRFMLP(nn.Module):

    def __init__(self, input_dim, hidden_dim=64, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )



    def forward(self, x):
        return self.net(x)