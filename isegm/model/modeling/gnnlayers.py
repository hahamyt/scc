import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class SimBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2 * dim, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, q):
        input = torch.cat([x, q.repeat(1, x.shape[1], 1)], dim=-1)
        return self.proj(input)          # 同时考虑大小和方向


class AdjBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()

    def forward(self, x):
        norm = torch.norm(x, dim=1, keepdim=True)  # 计算每个向量的范数，默认有自环，无需额外添加
        x_normalized = x / (norm + 1e-9)  # 防止除以零
        adj = torch.einsum('ik,jk->ij', x_normalized, x_normalized)
        return adj

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(y).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn_ = attn.softmax(dim=-1)
        attn = self.attn_drop(attn_)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_

def mean_query(x, mask):
    mask_int = mask.int()
    # 计算掩码加权的值
    weighted_x = x * mask_int
    # 计算掩码的非零元素总数
    non_zero_count = mask_int.sum(dim=1, keepdim=True)
    # 避免除以0的情况，如果有的话
    non_zero_count[non_zero_count == 0] = 1
    # 计算平均值
    query = weighted_x.sum(dim=1, keepdim=True) / non_zero_count
    return query