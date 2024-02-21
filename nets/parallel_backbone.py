"""
结合 maxvit,davit,twins,mobilenetv3,convnext等结构的模块

maxvit:         https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/maxxvit.py
davit:          https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/davit.py
twins:          https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/twins.py
mobilenetv3:    https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py
convnext:       https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py
"""

import math
from functools import partial
from typing import List

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from timm.layers import DropPath
from timm.layers import to_2tuple, _assert
from timm.models._features_fx import register_notrace_function


#---------------------------------------------------#
#   Conv+BN+Act
#---------------------------------------------------#
class ConvNormActivation(nn.Sequential):
    '''
    卷积,bn,激活函数
    '''
    def __init__(
        self,
        in_planes:   int,      # in_channel
        out_planes:  int,      # out_channel
        kernel_size: int = 3,
        stride:      int = 1,
        groups:      int = 1,
        norm_layer       = nn.BatchNorm2d,
        activation_layer = nn.ReLU
    ):
        # 这个padding仅适用于k为奇数的情况,偶数不适用
        padding = (kernel_size - 1) // 2

        super().__init__(nn.Conv2d(in_channels  =in_planes,
                                    out_channels=out_planes,
                                    kernel_size =kernel_size,
                                    stride      =stride,
                                    padding     =padding,
                                    groups      =groups,
                                    bias        =False),
                                    norm_layer(out_planes),
                                    activation_layer())


#---------------------------------------------------#
#   旧的写法,过程和新的一样
#   注意力机制
#       对特征矩阵每一个channel进行池化,得到长度为channel的一维向量,使用两个全连接层,
#       两个线性层的长度,最后得到权重,然后乘以每层矩阵的原值
#           线性层长度变化: channel -> channel / 4 -> channel
#---------------------------------------------------#
class SqueezeExcitation(nn.Module):
    def __init__(
        self,
        input_c: int,
        squeeze_factor: int = 4
    ):
        '''
        input_c: DW卷积输出通道个数(就是输入的通道个数)
        squeeze_factor: 中间层缩小倍数
        '''
        super().__init__()
        # 缩小4倍再调整为8的整数倍
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        # 两个卷积作为全连接层,kernel为1
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    def forward(self, x: Tensor) -> Tensor:
        '''
        x是特征矩阵
        '''
        # [b, channel, height, width] -> [batch, channel, 1, 1]
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        # 两个线性层的激活函数不同
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.sigmoid(scale)

        # [batch, channel, 1, 1] * [batch, channel, height, width]
        # 高维度矩阵相乘是最后两个维度相乘,所以是 [1, 1] 点乘 [h, w]
        return scale * x


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    将卷积核个数(输出通道个数)调整为最接近round_nearest的整数倍,就是8的整数倍,对硬件更加友好
    ch:      输出通道个数
    divisor: 奇数,必须将ch调整为它的整数倍
    min_ch:  最小通道数
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    # 最小为8
    if min_ch is None:
        min_ch = divisor
    # 调整到离8最近的值,类似于四舍五入
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


#---------------------------------------------------#
#   倒残差结构
#   残差:   两端channel多,中间channel少
#       降维 --> 升维
#   倒残差: 两端channel少,中间channel多
#       升维 --> 降维
#   1x1 3x3DWConv SE 1x1
#   最后的1x1Conv没有激活函数
#---------------------------------------------------#
class InvertedResidual(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        expand_ratio: float = 4.0,
        kernel: int = 3,            # kernel 3 or 5
        stride: int = 1,            # DW卷积步长
        use_se: bool = True,        # 注意力机制
        norm_layer = nn.BatchNorm2d,
        activation_layer = nn.GELU,
    ):
        super().__init__()

        input_c = _make_divisible(dim_in, 8)
        expanded_c = _make_divisible(int(dim_in * expand_ratio), 8)
        out_c = _make_divisible(dim_out, 8)

        # 判断每一层步长是否为1或2
        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        # 列表,每个元素都是Module类型
        layers: List[nn.Module] = []

        # 扩展维度
        layers.append(ConvNormActivation(input_c,                           # in_channel
                                        expanded_c,                         # out_channel
                                        kernel_size     =1,                 # 只变换维度
                                        norm_layer      =norm_layer,        # bn层
                                        activation_layer=activation_layer)) # 激活层

        # depthwise DW卷积
        layers.append(ConvNormActivation(expanded_c,  # in_channel == out_channel == group
                                        expanded_c,
                                        kernel_size     =kernel,
                                        stride          =stride,
                                        groups          =expanded_c,
                                        norm_layer      =norm_layer,        # bn层
                                        activation_layer=activation_layer)) # 激活层

        # 注意力机制
        if use_se:                      # 参数是上层DW输出的维度,efficientv1中是开始输入的维度
            layers.append(SqueezeExcitation(expanded_c))

        # project 最后的 1x1, 降维的卷积层 不使用激活函数
        layers.append(ConvNormActivation(expanded_c,
                                        out_c,
                                        kernel_size     =1,
                                        norm_layer      =norm_layer,
                                        activation_layer=nn.Identity))   # 线性激活,不做任何处理

        self.block        = nn.Sequential(*layers)  # 卷积层

        self.shortcut = ConvNormActivation(input_c, out_c, kernel_size=1, stride=stride) if (input_c != out_c) or stride != 1 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)

        result += self.shortcut(x)

        # 最后没有激活函数,是线性激活
        return result


class Permute(nn.Module):
    """This module returns a view of the tensor input with its dimensions permuted.

    Args:
        dims (List[int]): The desired ordering of dimensions
    """

    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return torch.permute(x, self.dims)


#-----------------------------------------------#
#   通道缩放
#   [B, H, W, C] * [C] = [B, H, W, C]
#-----------------------------------------------#
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma
        return x.mul_(gamma) if self.inplace else x * gamma


#-----------------------------------------------#
#   通道缩放
#   [B, C, H, W] * [1, C, 1, 1] = [B, C, H, W]
#-----------------------------------------------#
class LayerScale2d(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma.view(1, -1, 1, 1)
        return x.mul_(gamma) if self.inplace else x * gamma


class GroupNorm1(nn.GroupNorm):
    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(1, num_channels, **kwargs)


#------------------------------------#
#   7x7DWConv -> 1x1Conv -> 1x1Conv
#------------------------------------#
class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        stride: int = 1,
        layer_scale: float = 1e-5,
        norm_layer = GroupNorm1,
        activation_layer = nn.GELU,
    ) -> None:
        super().__init__()

        # 输入,输出dim的最大公约数
        groups = math.gcd(dim_in, dim_out)

        self.block = nn.Sequential(
            # DWConv
            nn.Conv2d(dim_in, dim_out, kernel_size=7, stride=stride, padding=3, groups=groups, bias=True),  # [B, C, H, W] -> [B, C, H, W]
            norm_layer(dim_out),
            nn.Conv2d(dim_out, 4 * dim_out, kernel_size=1, bias=True),                                      # [B, C, H, W] -> [B, 4*C, H, W]
            activation_layer(),
            nn.Conv2d(4 * dim_out, dim_out, kernel_size=1, bias=True),                                      # [B, 4*C, H, W] -> [B, C, H, W]
            LayerScale2d(dim_out, layer_scale),
        )

        self.shortcut = nn.Sequential(nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=stride), norm_layer(dim_out)) if (dim_in != dim_out) or stride != 1 else nn.Identity()

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        result += self.shortcut(input)
        return result


#-----------------------------------------#
#   [..., C] -> [..., n*C] -> [..., C]
#-----------------------------------------#
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer = nn.GELU,
        norm_layer = None,
        bias: bool = True,
        drop: float = 0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):   # mix channel
        x = self.fc1(x)     # [B, N, C] -> [B, N, n*C]
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)     # [B, N, n*C] -> [B, N, C]
        x = self.drop2(x)
        return x


#-------------------------------------#
#   1x1Conv代替全连接层
#   宽高不为1,不是注意力
#-------------------------------------#
class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    """
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.ReLU,
        norm_layer=None,
        bias=True,
        drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        #-------------------------------------#
        #   使用k=1的Conv代替两个全连接层
        #-------------------------------------#
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)     # [B, C, H, W] -> [B, n*C, H, W]
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)     # [B, n*C, H, W] -> [B, C, H, W]
        return x


#--------------------#
#   [B, H, W, C]
#--------------------#
class AttentionCl(nn.Module):
    """ Channels-last multi-head attention (B, ..., C) """

    def __init__(
        self,
        dim: int,
        head_dim: int = 32,
        bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.
    ):
        super().__init__()
        assert dim % head_dim == 0, f"dim {dim} should be divided by head_dim {head_dim}."
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [B*nh*nw, H, W, C] 简写 [B, H, W, C]
        Returns:
            torch.Tensor: [B*nh*nw, H, W, C]
        """
        B, H, W, C = x.shape                                        # [B, H, W, C]

        qkv = self.qkv(x)                                           # [B, H, W, C] -> [B, H, W, 3*C]
        qkv = qkv.view(B, -1, 3, self.num_heads, self.head_dim)     # [B, H, W, 3*C] -> [B, H*W, 3, h, c]  h * c = C
        qkv = qkv.permute(2, 0, 3, 1, 4)                            # [B, H*W, 3, h, c] -> [3, B, h, H*W, c]
        q, k, v = qkv.unbind(0)                                     # [3, B, h, H*W, c] -> 3 * [B, h, H*W, c]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)                              # [B, h, H*W, c] @ [B, h, c, H*W] = [B, h, H*W, H*W]
        attn = attn.softmax(dim=-1)                                 # 取每一列,在行上做softmax
        attn = self.attn_drop(attn)
        x = attn @ v                                                # [B, h, H*W, H*W] @ [B, h, H*W, c] = [B, h, H*W, c]

        x = x.transpose(1, 2)                   # [B, h, H*W, c] -> [B, H*W, h, c]
        x = x.reshape(B, H, W, C)               # [B, H*W, h, c] -> [B, H, W, C]

        x = self.proj(x)                        # [B, H, W, C] -> [B, H, W, C]
        x = self.proj_drop(x)
        return x


# windows
# [B, H, W, C] -> [B, nh, h, nw, w, C] -> [B, nh, nw, h, w, C]
# [B, nh, nw, h, w, C] -> [B*nh*nw, h, w, C]
# [B*nh*nw, h, w, C] -> [B*nh*nw, h, w, C]
# [B*nh*nw, h, w, C] -> [B, nh, nw, h, w, C]
# [B, nh, nw, h, w, C] -> [B, nh, h, nw, w, C] -> [B, H, W, C]
def window_partition(x, window_size: List[int]):
    """将宽高划分为不同的小格子
    [B, H, W, C] -> [B, nh, h, nw, w, C] -> [B, nh, nw, h, w, C] -> [B*nh*nw, h, w, C]

    Args:
        x (Tensor):              [B, H, W, C]
        window_size (List[int]): [h, w] nh * h = H  nw * w = W

    Returns:
        Tensor: [B*nh*nw, h, w, C]
    """
    B, H, W, C = x.shape
    _assert(H % window_size[0] == 0, f'height ({H}) must be divisible by window ({window_size[0]})')
    _assert(W % window_size[1] == 0, f'width ({W}) must be divisible by window ({window_size[1]})')
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)  # [B, H, W, C]         -> [B, nh, h, nw, w, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()                                          # [B, nh, h, nw, w, C] -> [B, nh, nw, h, w, C]
    windows = windows.view(-1, window_size[0], window_size[1], C)                               # [B, nh, nw, h, w, C] -> [B*nh*nw, h, w, C]
    return windows


@register_notrace_function  # reason: int argument is a Proxy
def window_reverse(windows, window_size: List[int], img_size: List[int]):
    """将windows数据还原
    [B*nh*nw, h, w, C] -> [B, nh, nw, h, w, C] -> [B, nh, h, nw, w, C] -> [B, H, W, C]

    Args:
        windows (Tensor):        [B*nh*nw, h, w, C]
        window_size (List[int]): [h, w] nh * h = H  nw * w = W
        img_size (List[int]):    [H, W]

    Returns:
        Tensor: [B, H, W, C]
    """
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)   # [B*nh*nw, h, w, C]   -> [B, nh, nw, h, w, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()                                                        # [B, nh, nw, h, w, C] -> [B, nh, h, nw, w, C]
    x = x.view(-1, H, W, C)                                                                             # [B, nh, h, nw, w, C] -> [B, H, W, C]
    return x


# grid
# [B, H, W, C] -> [B, h, nh, w, nw, C] -> [B, nh, nw, h, w, C]
# [B, nh, nw, h, w, C] -> [B*nh*nw, h, w, C]
# [B*nh*nw, h, w, C] -> [B*nh*nw, h, w, C]
# [B*nh*nw, h, w, C] -> [B, nh, nw, h, w, C]
# [B, nh, nw, h, w, C] -> [B, h, nh, w, nw, C] -> [B, H, W, C]
def grid_partition(x, grid_size: List[int]):
    """将宽高划分为不同的小格子,然后将相同位置的数据拼到一起
    [B, H, W, C] -> [B, h, nh, w, nw, C] -> [B, nh, nw, h, w, C] -> [B*nh*nw, h, w, C]

    Args:
        x (Tensor):            [B, H, W, C]
        grid_size (List[int]): [h, w] nh * h = H  nw * w = W

    Returns:
        Tensor: [B*nh*nw, h, w, C]
    """
    B, H, W, C = x.shape
    _assert(H % grid_size[0] == 0, f'height {H} must be divisible by grid {grid_size[0]}')
    _assert(W % grid_size[1] == 0, f'width {W} must be divisible by grid {grid_size[1]}')
    x = x.view(B, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1], C)          # [B, H, W, C]         -> [B, h, nh, w, nw, C]
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous()                                          # [B, h, nh, w, nw, C] -> [B, nh, nw, h, w, C]
    windows = windows.view(-1, grid_size[0], grid_size[1], C)                                   # [B, nh, nw, h, w, C] -> [B*nh*nw, h, w, C]
    return windows


@register_notrace_function  # reason: int argument is a Proxy
def grid_reverse(windows, grid_size: List[int], img_size: List[int]):
    """将网格数据还原
    [B*nh*nw, h, w, C]   -> [B, nh, nw, h, w, C] -> [B, h, nh, w, nw, C] -> [B, H, W, C]

    Args:
        windows (Tensor):      [B*nh*nw, h, w, C]
        grid_size (List[int]): [h, w] nh * h = H  nw * w = W
        img_size (List[int]):  [H, W]

    Returns:
        Tensor: [B, H, W, C]
    """
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // grid_size[0], W // grid_size[1], grid_size[0], grid_size[1], C)   # [B*nh*nw, h, w, C]   -> [B, nh, nw, h, w, C]
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()                                                # [B, nh, nw, h, w, C] -> [B, h, nh, w, nw, C]
    x = x.view(-1, H, W, C)                                                                     # [B, h, nh, w, nw, C] -> [B, H, W, C]
    return x


#-------------------------------------------------------------------------------#
#   ChannelBlock中用的ChannelAttention  !!!使用了通道的注意力!!! edgenext也用了这个
#-------------------------------------------------------------------------------#
class ChannelAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        head_dim: int = 32,
        bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.
    ):
        super().__init__()
        assert dim % head_dim == 0, f"dim {dim} should be divided by head_dim {head_dim}."
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor):
        B, H, W, C = x.shape

        qkv = self.qkv(x)                                               # [B, H, W, C] -> [B, H, W, 3*C]
        qkv = qkv.reshape(B, -1, 3, self.num_heads, self.head_dim)      # [B, H, W, 3*C] -> [B, H*W, 3, h, c]   C = h * c
        qkv = qkv.permute(2, 0, 3, 1, 4)                                # [B, H*W, 3, h, c] -> [3, B, h, H*W, c]
        q, k, v = qkv.unbind(0)                                         # [3, B, h, H*W, c] -> 3 * [B, h, H*W, c]

        k = k * self.scale
        attention = k.transpose(-1, -2) @ v     # [B, h, c, H*W] * [B, h, H*W, c] = [B, h, c, c]    !!!使用了通道的注意力!!! edgenext也用了这个
        attention = attention.softmax(dim=-1)   # 行上做softmax
        attention = self.attn_drop(attention)

        x = (attention @ q.transpose(-1, -2))   # [B, h, c, c] @ [B, h, c, H*W] = [B, h, c, H*W]
        x = x.permute(0, 3, 1, 2)               # [B, h, c, H*W] -> [B, H*W, h, c]
        x = x.reshape(B, H, W, C)               # [B, H*W, h, c] -> [B, H, W, C]

        x = self.proj(x)                        # [B, H, W, C] -> [B, H, W, C]
        x = self.proj_drop(x)
        return x


#---------------------------------------------------------#
#   q = q(x)
#   kv = kv(conv(x)) conv是patch,降低x的宽高,提取全局特征
#---------------------------------------------------------#
class GlobalSubSampleAttn(nn.Module):
    """ GSA: using a  key to summarize the information for a group to be efficient.
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
        self,
        dim: int,
        head_dim: int = 32,
        bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        sr_ratio: int = 1
    ):
        super().__init__()
        assert dim % head_dim == 0, f"dim {dim} should be divided by head_dim {head_dim}."

        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=bias)
        self.kv = nn.Linear(dim, dim * 2, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            #---------------------------------------------------------#
            #   对x处理,用来当做 key value 的输入
            #   kernel和stride相同 x的HW总为7,因为sr_ratio为 8 4 2 1 最后的1不处理
            #---------------------------------------------------------#
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x):
        """
        Args:
            x (Tensor): [B, N, C]

        Returns:
            Tensor: [B, N, C]
        """
        B, H, W, C = x.shape   # [B, H, W, C]

        # 只计算query  h * c = C
        # [B, H, W, C] -> [B, H, W, C] -> [B, H*W, h, c]    pcpvt每个head的channel总为64,svt中为32
        q = self.q(x).reshape(B, -1, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)   # [B, H*W, h, c] -> [B, h, H*W, c]

        #--------------------------------------------#
        #   对x处理,减少宽高,用来当做 key value 的输入
        #--------------------------------------------#
        if self.sr is not None:
            x = x.permute(0, 3, 1, 2)   # [B, H, W, C] -> [B, C, H, W]

            x = self.sr(x)              # [B, C, H, W] -> [B, C, 7, 7]
            x = x.reshape(B, C, -1)     # [B, C, 7, 7] -> [B, C, 49]
            x = x.permute(0, 2, 1)      # [B, C, 49] -> [B, 49, C]
            x = self.norm(x)

        kv = self.kv(x)                                                 # [B, 49, C] -> [B, 49, 2*C]
        kv = kv.reshape(B, -1, 2, self.num_heads, C // self.num_heads)  # [B, 49, 2*C] -> [B, 49, 2, h, c]
        kv = kv.permute(2, 0, 3, 1, 4)                                  # [B, 49, 2, h, c] -> [2, B, h, 49, c]
        k, v = kv.unbind(0)                                             # [2, B, h, 49, c] -> 2 * [B, h, 49, c]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # [B, h, H*W, c] @ [B, h, c, 49] = [B, h, H*W, 49]
        attn = attn.softmax(dim=-1)     # 取每一列,在行上做softmax
        attn = self.attn_drop(attn)
        x = attn @ v                    # [B, h, H*W, 49] @ [B, h, 49, c] = [B, h, H*W, c]

        x = x.transpose(1, 2)           # [B, h, H*W, c] -> [B, H*W, h, c]
        x = x.reshape(B, H, W, C)       # [B, H*W, h, c] -> [B, H, W, C]

        x = self.proj(x)                # [B, H, W, C] -> [B, H, W, C]
        x = self.proj_drop(x)

        return x


#-----------------------------------------------------------------#
#                           in
#                            │
#   ┌───────────┬────────────┼────────────┬─────────────┐
#   │           │            │            │             │
#   │      windows_attn  grid_attn  channel_attn  subsample_attn
#   │           │            │            │             │
#   └───────────┴─────────── + ───────────┴─────────────┘
#                            │
#                           out
#-----------------------------------------------------------------#
class ParallelAttentionCl(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        window_size: int = 10,
        grid_size: int = 10,
        sr_ratio: int = 1,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        drop_path: float = 0.,
        norm_layer = nn.LayerNorm,
        layer_scale: float = 1e-5,
        attns: list[bool] = [True, True, False, False], # [windows_attn, grid_attn, channel_attn, subsample_attn]
    ):
        super().__init__()
        self.attns = attns

        self.window_size = to_2tuple(window_size)
        self.grid_size = to_2tuple(grid_size)

        #---------- windows attn ----------#
        if self.attns[0]:
            self.norm_window = norm_layer(dim)
            self.attn_window = AttentionCl(
                dim=dim,
                head_dim=head_dim,
                bias=True,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
            self.ls_window = LayerScale(dim, layer_scale)
            self.drop_path_window = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #---------- windows attn ----------#

        #---------- grid attn ----------#
        if self.attns[1]:
            self.norm_grid = norm_layer(dim)
            self.attn_grid = AttentionCl(
                dim=dim,
                head_dim=head_dim,
                bias=True,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
            self.ls_grid = LayerScale(dim, layer_scale)
            self.drop_path_grid = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #---------- grid attn ----------#

        #---------- channel attn ----------#
        if self.attns[2]:
            self.norm_channel = norm_layer(dim)
            self.attn_channel = ChannelAttention(
                dim=dim,
                head_dim=head_dim,
                bias=True,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
            self.ls_channel = LayerScale(dim, layer_scale)
            self.drop_path_channel = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #---------- channel attn ----------#

        #---------- subsample attn ----------#
        if self.attns[3]:
            self.norm_subsample = norm_layer(dim)
            self.attn_subsample = GlobalSubSampleAttn(
                dim=dim,
                head_dim=head_dim,
                bias=True,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                sr_ratio=sr_ratio,
            )
            self.ls_subsample = LayerScale(dim, layer_scale)
            self.drop_path_subsample = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #---------- subsample attn ----------#

    def forward(self, x):
        B, H, W, C = x.shape
        img_size = [H, W]
        xs = [x]

        #---------- windows attn ----------#
        if self.attns[0]:
            windows = self.norm_window(x)
            windows = window_partition(windows, self.window_size)
            windows = self.attn_window(windows)
            windows = window_reverse(windows, self.window_size, img_size)
            windows = self.ls_window(windows)
            windows = self.drop_path_window(windows)
            xs.append(windows)
        #---------- windows attn ----------#

        #---------- grid attn ----------#
        if self.attns[1]:
            grids = self.norm_grid(x)
            grids = grid_partition(grids, self.grid_size)
            grids = self.attn_grid(grids)
            grids = grid_reverse(grids, self.grid_size, img_size)
            grids = self.ls_grid(grids)
            grids = self.drop_path_grid(grids)
            xs.append(grids)
        #---------- grid attn ----------#

        #---------- channel attn ----------#
        if self.attns[2]:
            channels = self.norm_channel(x)
            channels = self.attn_channel(channels)
            channels = self.ls_channel(channels)
            channels = self.drop_path_channel(channels)
            xs.append(channels)
        #---------- channel attn ----------#

        #---------- subsample attn ----------#
        if self.attns[3]:
            subsample = self.norm_subsample(x)
            subsample = self.attn_subsample(subsample)
            subsample = self.ls_subsample(subsample)
            subsample = self.drop_path_subsample(subsample)
            xs.append(subsample)
        #---------- subsample attn ----------#

        x = sum(xs)

        return x


def test_parallel_attentionCl():
    x = torch.ones((1, 512, 80, 80))
    model = ParallelAttentionCl(dim=512, head_dim=32, window_size=5, grid_size=5).eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size())
    if False:
        torch.onnx.export(
            model=model,
            args=x,
            f="ParallelAttentionCl.onnx",
            opset_version=17
        )


#----------------------------------------#
#             in
#              │
#            conv(include residual)
#              │
#            attns(include residual)
#              │
#       ┌──────┤
#       │      │
#       │     mlp
#       │      │
#       └───── +
#              │
#             out
#----------------------------------------#
class ParallelBlock(nn.Module):
    def __init__(
        self,
        conv_type: str = "mobilenetv3", # mobilenetv3, convnext
        dim_in: int = 512,
        dim_out: int = 512,
        head_dim: int = 32,
        stride: int = 1,
        window_size: int = 10,
        grid_size: int = 10,
        sr_ratio: int = 1,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        mlp_drop: float = 0.,
        drop_path: float = 0.,
        act_layer = nn.GELU,
        norm_layer = nn.LayerNorm,
        layer_scale: float = 1e-5,
        attns: list[bool] = [True, True, False, False], # [windows_attn, grid_attn, channel_attn, subsample_attn]
    ):
        super().__init__()
        #--------- conv ----------#
        assert conv_type in ["mobilenetv3", "convnext"]
        if conv_type == "mobilenetv3":
            self.conv = InvertedResidual(dim_in=dim_in, dim_out=dim_out, kernel=3, stride=stride, use_se=True, activation_layer=act_layer)
        elif conv_type == "convnext":
            self.conv = ConvNeXtBlock(dim_in=dim_in, dim_out=dim_out, stride=stride, layer_scale=layer_scale, activation_layer=act_layer)
        #--------- conv ----------#

        #--------- attn ----------#
        self.attn = ParallelAttentionCl(
            dim_out,
            head_dim=head_dim,
            window_size=window_size,
            grid_size=grid_size,
            sr_ratio=sr_ratio,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            layer_scale=layer_scale,
            attns=attns,
        )
        #--------- attn ----------#

        #---------- mlp ----------#
        self.norm_mlp = norm_layer(dim_out)
        self.mlp = Mlp(
            in_features=dim_out,
            hidden_features=int(dim_out * 4),
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop=mlp_drop,
        )
        self.ls_mlp = LayerScale(dim_out, layer_scale)
        self.drop_path_mlp = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #---------- mlp ----------#

    def forward(self, x):
        #--------- conv ----------#
        x = self.conv(x)
        #--------- conv ----------#

        x = x.permute(0, 2, 3, 1)   # [B, C, H, W] -> [B, H, W, C]

        #--------- attn ----------#
        x = self.attn(x)
        #--------- attn ----------#

        #---------- mlp ----------#
        x = x + self.drop_path_mlp(self.ls_mlp(self.mlp(self.norm_mlp(x))))
        #---------- mlp ----------#

        x = x.permute(0, 3, 1, 2)   # [B, H, W, C] -> [B, C, H, W]

        return x


def test_parallel_block():
    x = torch.ones((1, 256, 80, 80))
    model = ParallelBlock(conv_type="mobilenetv3", dim_in=256, dim_out=512, head_dim=32, stride=2).eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size()) # [1, 512, 40, 40]


class ParallelStage(nn.Module):
    def __init__(
        self,
        depth: int = 3,
        conv_type: str = "mobilenetv3",
        dim_in: int = 512,
        dim_out: int = 512,
        head_dim: int = 32,
        stride: int = 1,
        window_size: int = 10,
        grid_size: int = 10,
        sr_ratio: int = 1,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        mlp_drop: float = 0.,
        drop_path: float = 0.,
        act_layer = nn.GELU,
        norm_layer = nn.LayerNorm,
        layer_scale: float = 1e-5,
        attns: list[bool] = [True, True, False, False], # [windows_attn, grid_attn, channel_attn, subsample_attn]
    ):
        super().__init__()
        blocks = []
        for i in range(depth):
            blocks.append(
                ParallelBlock(
                    conv_type=conv_type,
                    dim_in=dim_in if i == 0 else dim_out,
                    dim_out=dim_out,
                    head_dim=head_dim,
                    stride=stride if i == 0 else 1,
                    window_size=window_size,
                    grid_size=grid_size,
                    sr_ratio=sr_ratio,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    mlp_drop=mlp_drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    layer_scale=layer_scale,
                    attns=attns,
                ))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.blocks(x)
        return x


def test_parallel_stage():
    x = torch.ones((1, 256, 80, 80))
    model = ParallelStage(depth=3, conv_type="mobilenetv3", dim_in=256, dim_out=512, head_dim=32, stride=2).eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size()) # [1, 512, 40, 40]


def autopad(k, p=None, d=1):
    # kernel, padding, dilation
    # 对输入的特征层进行自动padding，按照Same原则
    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class SiLU(nn.Module):
    # SiLU激活函数
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Conv(nn.Module):
    # 标准卷积+标准化+激活函数
    default_act = SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act    = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Backbone(nn.Module):
    def __init__(
        self,
        base_channels: int,
        base_depth: int,
        deep_mul: float,
        conv_type: str = "mobilenetv3", # mobilenetv3, convnext
        window_size: int = 10,
        grid_size: int = 10,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        mlp_drop: float = 0.,
        drop_path: float = 0.,
        sr_ratio: list[int] = [8, 4, 2, 1], # GlobalSubSampleAttn ratio
        attns: list[bool] = [True, True, False, False], # [windows_attn, grid_attn, channel_attn, subsample_attn]
        last_attn: bool = False,
    ):
        super().__init__()
        #-----------------------------------------------#
        #   输入图片是3, 640, 640
        #-----------------------------------------------#
        # 3, 640, 640 => 64, 320, 320
        self.stem = Conv(3, base_channels, 3, 2)

        p_stage = partial(
            ParallelStage,
            conv_type=conv_type,
            head_dim=base_channels,
            stride=2,
            window_size=window_size,
            grid_size=grid_size,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_drop=mlp_drop,
            drop_path=drop_path,
            attns=attns,
        )

        # 64, 320, 320 => 128, 160, 160 => 128, 160, 160
        # self.dark2 = nn.Sequential(
        #     Conv(base_channels, base_channels * 2, 3, 2),
        #     C2f(base_channels * 2, base_channels * 2, base_depth, True),
        # )
        self.dark2 = p_stage(
            depth=base_depth,
            dim_in=base_channels,
            dim_out=base_channels * 2,
            sr_ratio=sr_ratio[0],
        )

        # 128, 160, 160 => 256, 80, 80 => 256, 80, 80
        # self.dark3 = nn.Sequential(
        #     Conv(base_channels * 2, base_channels * 4, 3, 2),
        #     C2f(base_channels * 4, base_channels * 4, base_depth * 2, True),
        # )
        self.dark3 = p_stage(
            depth=base_depth * 2,
            dim_in=base_channels * 2,
            dim_out=base_channels * 4,
            sr_ratio=sr_ratio[1],
        )

        # 256, 80, 80 => 512, 40, 40 => 512, 40, 40
        # self.dark4 = nn.Sequential(
        #     Conv(base_channels * 4, base_channels * 8, 3, 2),
        #     C2f(base_channels * 8, base_channels * 8, base_depth * 2, True),
        # )
        self.dark4 = p_stage(
            depth=base_depth * 2,
            dim_in=base_channels * 4,
            dim_out=base_channels * 8,
            sr_ratio=sr_ratio[2],
        )

        # 512, 40, 40 => 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 20, 20
        # self.dark5 = nn.Sequential(
        #     Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2),
        #     C2f(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth, True),
        #     SPPF(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), k=5)
        # )
        dark5 = [
            p_stage(
                depth=base_depth,
                dim_in=base_channels * 8,
                dim_out=int(base_channels * 16 * deep_mul),
                sr_ratio=sr_ratio[3],
            )
        ]
        if last_attn:
            dark5.extend([
                Permute([0, 2, 3, 1]),  # [B, C, H, W] -> [B, H, W, C]
                AttentionCl(
                    dim=int(base_channels * 16 * deep_mul),
                    head_dim=base_channels,
                ),
                Permute([0, 3, 1, 2]),  # [B, H, W, C] -> [B, C, H, W]
            ])
        self.dark5 = nn.Sequential(*dark5)

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        #-----------------------------------------------#
        #   dark3的输出为256, 80, 80，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark3(x)
        feat1 = x
        #-----------------------------------------------#
        #   dark4的输出为512, 40, 40，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark4(x)
        feat2 = x
        #-----------------------------------------------#
        #   dark5的输出为1024 * deep_mul, 20, 20，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark5(x)
        feat3 = x

        #-----------------------------------------------#
        #   feat1: 256, 80, 80
        #   feat2: 512, 40, 40
        #   feat3: 1024 * deep_mul, 20, 20
        #-----------------------------------------------#
        return feat1, feat2, feat3


def test_backbone():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    phi = "n"
    depth_dict          = {'n' : 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.00,}
    width_dict          = {'n' : 0.25, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
    deep_width_dict     = {'n' : 1.00, 's' : 1.00, 'm' : 0.75, 'l' : 0.50, 'x' : 0.50,}
    dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]

    base_channels       = int(wid_mul * 64)  # 64
    base_depth          = max(round(dep_mul * 3), 1)  # 3

    print(base_channels, base_depth, deep_mul, int(base_channels * 16 * deep_mul))
    # n: 16 1 1.0  256
    # s: 32 1 1.0  512
    # m: 48 2 0.75 576
    # l: 64 3 0.5  512
    # x: 80 3 0.5  640

    model = Backbone(
        base_channels,
        base_depth,
        deep_mul,
        conv_type="convnext",
        window_size=10,
        grid_size=10,
        sr_ratio=[8, 4, 2, 1],
        attns=[True, True, True, True],
    ).to(device)
    x = torch.ones(1, 3, 640, 640).to(device)

    model.eval()
    with torch.inference_mode():
        feats = model(x)
    for feat in feats:
        print(feat.size())

    if False:
        onnx_path = f"backbone-{phi}.onnx"
        torch.onnx.export(
            model=model,
            args=x,
            f=onnx_path,
            input_names=['image'],
            output_names=['feat1', 'feat2', 'feat3'],
            opset_version=17,
        )
        import onnx
        from onnxsim import simplify

        # 载入onnx模型
        model_ = onnx.load(onnx_path)

        # 简化模型
        model_simple, check = simplify(model_)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simple, onnx_path)
        print('finished exporting ' + onnx_path)


if __name__ == "__main__":
    # test_partition_attentionCl()
    # test_parallel_attentionCl()
    # test_parallel_block()
    # test_parallel_stage()
    test_backbone()
