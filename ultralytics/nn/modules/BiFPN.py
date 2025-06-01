
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50

__all__ = ["BiFPN_Concat", "BiFPN"]


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class BiFPN_Concat(nn.Module):
    def __init__(self, c1, c2):
        super(BiFPN_Concat, self).__init__()
        self.w1_weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2_weight = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = Conv(c1, c2, 1, 1, 0)
        self.act = nn.ReLU()

    def forward(self, x):
        if len(x) == 2:
            w = self.w1_weight
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = self.conv(self.act(weight[0] * x[0] + weight[1] * x[1]))
        elif len(x) == 3:
            w = self.w2_weight
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = self.conv(self.act(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]))
        return x



class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class BiFPN(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(length, dtype=torch.float32), requires_grad=True)
        self.swish = swish()
        self.epsilon = 0.0001

    def forward(self, x):
        weights = self.weight / (torch.sum(self.swish(self.weight), dim=0) + self.epsilon)
        weighted_feature_maps = [weights[i] * x[i] for i in range(len(x))]
        stacked_feature_maps = torch.stack(weighted_feature_maps, dim=0)
        result = torch.sum(stacked_feature_maps, dim=0)
        return result


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mhsa = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)  # (B, Seq_len, Dim)
        attn_output, _ = self.mhsa(x, x, x)
        attn_output = self.norm(attn_output + x)  # Residual Connection
        return attn_output.permute(0, 2, 1).view(b, c, h, w)  # Chuyển về lại


class BiFPN_Transformer(nn.Module):
    def __init__(self, length, embed_dim=128, num_heads=4):
        super().__init__()
        self.length = length
        self.weight = nn.Parameter(torch.ones(length, dtype=torch.float32), requires_grad=True)
        self.epsilon = 1e-4  # Giá trị epsilon nhỏ hơn
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)

    def forward(self, x):
        device = x[0].device  # Lấy thiết bị của tensor đầu vào
        weights = self.weight.to(device)  # Chuyển weight lên đúng thiết bị
        norm_weights = weights / (torch.sum(F.silu(weights), dim=0) + self.epsilon)

        weighted_feature_maps = [norm_weights[i] * x[i] for i in range(self.length)]
        stacked_feature_maps = torch.stack(weighted_feature_maps, dim=0)
        result = torch.sum(stacked_feature_maps, dim=0)

        result = self.attention(result)  # Áp dụng Multi-Head Self-Attention
        return result
