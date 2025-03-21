
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50

__all__ = ["BiFPN_Concat", "BiFPN"]

from sympy.physics.paulialgebra import epsilon


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
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
    def __init__(self, dimension=1, num_inputs=2, use_relu=True):
        super(BiFPN_Concat, self).__init__()
        self.d = dimension
        self.num_inputs = num_inputs
        self.use_relu = use_relu # Cho phép lựa chọn dùng ReLU hay không

        # Khởi tạo trọng số học được
        self.w = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32), requires_grad=True)
        self.epsilon = 1e-4

    def forward(self, x):
        # Kiểm tra đầu vào chặt chẽ
        if not isinstance(x, list) or len(x) != self.num_inputs:
            raise ValueError(
                f"Input must be a list of {self.num_inputs} tensors. "
                f"Received: {type(x)} with length {len(x)}"
            )

        # Xử lý trọng số (ReLU tùy chọn)
        w = self.w
        if self.use_relu:
            w = F.relu(w)  # Đảm bảo trọng số không âm

        # Chuẩn hóa trọng số
        weight = w / (torch.sum(w, dim=0) + self.epsilon)

        # Áp dụng trọng số và concatenate
        weighted_features = [weight[i] * x[i] for i in range(self.num_inputs)]
        return torch.cat(weighted_features, dim=self.d)

# class BiFPN_Concat(nn.Module):
#     def __init__(self, dimension=1):
#         super(BiFPN_Concat, self).__init__()
#         self.d = dimension
#         self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
#         self.epsilon = 0.0001
#
#     def forward(self, x):
#             w = self.w
#             weight = w / (torch.sum(w, dim=0) + self.epsilon)
#             x = [weight[0] * x[0], weight[1] * x[1]]
#             return torch.cat(x, self.d)


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
