import torch
import torch.nn as nn
import torch.nn.functional as F


class DownsampleMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim // 2)

    def forward(self, x):
        return self.fc(x)


class UpsampleMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim * 2)

    def forward(self, x):
        return self.fc(x)

class MLPBlock(nn.Module):
    '''
        Linear --> LayerNorm --> Mish
    '''

    def __init__(self, inp_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(inp_dim, out_dim),
            nn.LayerNorm(out_dim),  # LayerNorm normalizes across the feature dimension
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)

