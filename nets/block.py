# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
if __name__ == "__main__":
    import sys
    sys.path.append("../")
from backbone import Conv, Bottleneck


#-------------------------------#
#   MaxSigmoidAttnBlock: 通过 guide 更新 image 的特征
#
#            image    guide
#              │        │
#     ┌────────┤        │
#     │        │        │
#    Conv     Conv    Linear
#     │        │        │
#     │        └─ attn ─┘
#     │             │
#     │            max
#     │             │
#     │          sigmoid
#     │             │
#     └───── * ─────┘
#            │
#           out
#-------------------------------#
class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments.
        arguments: ch_in, ch_out, num heads, embed channels, guide channels, layer scale.
        """
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape   # [B, 256, 40, 40]

        # text guide
        guide = self.gl(guide)                          # [B, 100, 512] -> [B, 100, 256]
        guide = guide.view(bs, -1, self.nh, self.hc)    # [B, 100, 256] -> [B, 100, 8, 32]
        # image
        embed = self.ec(x) if self.ec is not None else x# [B, 256, 40, 40] -> [B, 256, 40, 40]
        embed = embed.view(bs, self.nh, self.hc, h, w)  # [B, 8, 32, 40, 40]

        # attn: 查询的是每个head中的像素与每个文本相似度的最大值,将最大值通过sigmoid得到权重
        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)# [B, 8, 32, 40, 40] einsum [B, 100, 8, 32] = [B, 8, 40, 40, 100]
        aw = aw.max(dim=-1)[0] # [0] get value      # [B, 8, 40, 40, 100] get [B, 8, 40, 40]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]    # [B, 8, 40, 40] + [1, 8, 1, 1] = [B, 8, 40, 40]
        aw = aw.sigmoid() * self.scale              # 使用sigmoid

        # 经过sigmoid的attn直接和原像素进行逐位置相乘
        x = self.proj_conv(x)               # [B, 256, 40, 40] -> [B, 256, 40, 40]
        x = x.view(bs, self.nh, -1, h, w)   # [B, 256, 40, 40] -> [B, 8, 32, 40, 40]
        x = x * aw.unsqueeze(2)             # [B, 8, 32, 40, 40] * ([B, 8, 40, 40] -> [B, 8, 1, 40, 40]) = [B, 8, 32, 40, 40]
        return x.view(bs, -1, h, w)         # [B, 8, 32, 40, 40] -> [B, 256, 40, 40]    最后获取的是通过文本更新的图像特征


def test_max_sigmoid_attn_block():
    model = MaxSigmoidAttnBlock(c1=256, c2=256, nh=8, ec=256, gc=512).eval()
    x = torch.randn(1, 256, 40, 40)
    guide = torch.randn(1, 100, 512)
    with torch.inference_mode():
        y = model(x, guide)
    print(y.shape)  # [1, 256, 40, 40]


#----------------------------------------------------------------------#
#   C2fAttn: 提取图像特征和使用 guide 更新 image 的特征
#
#          image                                               guide
#            │                                                   │
#         cv1(1x1)                                               │
#            │                                                   │
#   ┌───── split ─────┐                                          │
#   │                 ├─────────┐                                │
#   │                 │  m_1(Bottleneck)                         │
#   │                 │         ├────────┐                       │
#   │                 │         │ m_2(Bottleneck)                │
#   │                 │         │        ├─────────┐             │
#   │                 │         │        │  m_3(Bottleneck)...   │
#   │                 │         │        │         ├────────────┐│
#   │                 │         │        │         │           Attn
#   └──── concat ─────┴─────────┴────────┴─────────┴────────────┘
#            │
#         cv2(1X1)
#            │
#           out
#----------------------------------------------------------------------#
class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initialize C2fAttn layer
        arguments: ch_in, ch_out, repeats, embed channels, num heads, guide channels, shortcut, groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, nh=nh, ec=ec, gc=gc)

    def forward(self, x, guide):
        # 进行一个卷积，然后划分成两份，每个通道都为c
        y = list(self.cv1(x).split((self.c, self.c), 1))
        # 每进行一次残差结构都保留，然后堆叠在一起，密集残差
        y.extend(m(y[-1]) for m in self.m)
        # 将最后一个残差的输出与文本特征融合，得到最终的融合特征
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


def test_c2f_attn():
    # 1024+512, 512, 3, 256, 8
    model = C2fAttn(c1=1024+512, c2=512, n=3, ec=256, nh=8).eval()
    x = torch.randn(1, 1024+512, 40, 40)
    guide = torch.randn(1, 100, 512)
    with torch.inference_mode():
        y = model(x, guide)
    print(y.shape)  # [1, 512, 40, 40]


#------------------------------------#
#   ImagePoolingAttn
#   通过文本和图片的注意力更新文本特征
#
#               images       text
#                 │           │
#  3 stage iamges │           │
#         ┌───── get ─────┐   │
#         │       │       │   │
#        Conv    Conv    Conv │
#         │       │       │   │
#        Pool    Pool    Pool │
#         │       │       │   │
#         └───── cat ─────┘   │
#                 │           │
#             ┌───┴───┐       ├───┐
#             │       │       │   │
#          k_proj  v_proj  q_proj │
#             │       │       │   │
#             └──── attn ─────┘   │
#                     │           │
#                  o_proj         │
#                     │           │
#                     └──── + ────┘
#                           │
#                          out
#------------------------------------#
class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments.
        embed channels, (multi stage channels), text channels, num heads, adaptive output size, layer scale.
        """
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0

        # 对多个stage的图像调整通道和大小
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])

        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]    # [B, C, h, w] -> [B, 256, 3, 3] -> [B, 256, 9] repeat 3 times
        x = torch.cat(x, dim=-1).transpose(1, 2)    # [B, 256, 9] * 3 cat = [B, 256, 27] -> [B, 27, 256]

        q = self.query(text)    # [B, 100, 512] -> [B, 100, 256]    query from text
        k = self.key(x)         # [B, 27, 256] -> [B, 27, 256]      key from image
        v = self.value(x)       # [B, 27, 256] -> [B, 27, 256]      value from image

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc) # [B, 100, 256] -> [B, 100, 8, 32]
        k = k.reshape(bs, -1, self.nh, self.hc) # [B, 27, 256] -> [B, 27, 8, 32]
        v = v.reshape(bs, -1, self.nh, self.hc) # [B, 27, 256] -> [B, 27, 8, 32]

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)  # [B, 100, 8, 32] einsum [B, 27, 8, 32] = [B, 100, 8, 27]
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)  # [B, 100, 8, 27] einsum [B, 27, 8, 32] -> [B, 100, 8, 32]
        x = self.proj(x.reshape(bs, -1, self.ec))   # [B, 100, 8, 32] -> [B, 100, 256] -> [B, 100, 512]
        return x * self.scale + text                # [B, 100, 512] * number + [B, 100, 512] = [B, 100, 512]    最终特征是通过图像更新的文本特征


def test_image_pooling_attn():
    model = ImagePoolingAttn(ec=256, ch=(256, 512, 1024)).eval()
    x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
    text = torch.randn(1, 100, 512)
    with torch.inference_mode():
        y = model(x, text)
    print(y.shape)  # [1, 100, 512]


class ContrastiveHead(nn.Module):
    """Contrastive Head for YOLO-World compute the region-text scores according to the similarity between image and text
    features.
    """

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        self.bias = nn.Parameter(torch.zeros([]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)          # [B, C, H, W]
        w = F.normalize(w, dim=-1, p=2)         # [B, K, C]
        x = torch.einsum("bchw,bkc->bkhw", x, w)# [B, C, H, W] einsum [B, K, C] -> [B, K, H, W]
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-Worldv2 using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        self.bias = nn.Parameter(torch.zeros([]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)                        # [B, C, H, W]
        w = F.normalize(w, dim=-1, p=2)         # [B, K, C]
        x = torch.einsum("bchw,bkc->bkhw", x, w)# [B, C, H, W] einsum [B, K, C] -> [B, K, H, W]
        return x * self.logit_scale.exp() + self.bias


def test_contrastive_head():
    chead = ContrastiveHead().eval()
    bnchead = BNContrastiveHead(512).eval()
    x = torch.randn(1, 512, 40, 40)
    w = torch.randn(1, 100, 512)
    with torch.inference_mode():
        y1 = chead(x, w)
        y2 = bnchead(x, w)
    print(y1.shape, y2.shape)   # [1, 100, 40, 40]


if __name__ == "__main__":
    test_max_sigmoid_attn_block()
    test_c2f_attn()
    test_image_pooling_attn()
    test_contrastive_head()
