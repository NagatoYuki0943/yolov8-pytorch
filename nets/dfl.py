import torch
from torch import Tensor
from torch import nn


class DFL(nn.Module):
    """
    DFL模块
    Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391 https://arxiv.org/abs/2006.04388
    使用卷积的操作进行相乘和相加

    预测结果取softmax   0.0	 0.1  0.0  0.0  0.4  0.5  0.0  0.0
    参考的固定值         0    1    2    3    4    5    6    7   conv中的权重
    点乘               0.1 * 1 + 0.4 * 4 + 0.5 * 5 = 4.2
    """
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.in_channels = in_channels
        #-----------------------------------------------------#
        #   不训练这个conv,权重设置为 [0...15], 直接和预测值相乘
        #   out_channels = 1,含义为计算结果channel为1
        #-----------------------------------------------------#
        self.conv = nn.Conv2d(in_channels, 1, 1, bias=False).requires_grad_(False)
        weight = torch.arange(in_channels, dtype=torch.float).reshape(1, in_channels, 1, 1)
        self.conv.weight.data[:] = weight

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        # [B, 16, H, W] -> [B, 1, H, W] -> [B, H, W]
        return self.conv(x.softmax(dim=1)).reshape(B, H, W)


if __name__ == "__main__":
    # 1代表batch, 4代表xyxy, 8400代表8400个框
    # x = torch.ones(1, 16, 4, 8400)
    x = torch.ones(1, 16, 1, 1)
    dfl = DFL(in_channels=16)
    for name, parameters in dfl.named_parameters():
        print(name, "=>", parameters.requires_grad)
    # conv.weight => False

    dfl.train()
    print(dfl(x).shape) # [1, 1, 1]
    print(dfl(x))       # [[[7.5000]]]
    print((torch.arange(16, dtype=torch.float).reshape(1, 16, 1, 1) * x.softmax(dim=1)).sum())
    # 7.5000
