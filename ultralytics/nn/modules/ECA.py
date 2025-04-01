import torch
from torch import nn

class ECA(nn.Module):
    def __init__(self, channels: int, k_size: int = 3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply global average pooling
        y = self.avg_pool(x)

        # Reshape and apply 1D convolution
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Apply sigmoid activation and element-wise multiplication
        return x * self.sigmoid(y)
