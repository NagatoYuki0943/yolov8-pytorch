import torch
from torch import nn
from torchvision import models


class Swin(nn.Module):
    def __init__(self, variance: str = "swin_t", pretrained=True) -> None:
        """swin swin_v2 backbone

        Args:
            variance (str, optional): swin version in [swin_t, swin_s, swin_b, swin_v2_t, swin_v2_s, swin_v2_b]. Defaults to "swin_t".
        """
        super().__init__()
        if variance == "swin_t":
            self.model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT if pretrained else None)
        elif variance == "swin_s":
            self.model = models.swin_s(weights=models.Swin_S_Weights.DEFAULT if pretrained else None)
        elif variance == "swin_b":
            self.model = models.swin_b(weights=models.Swin_B_Weights.DEFAULT if pretrained else None)
        elif variance == "swin_v2_t":
            self.model = models.swin_v2_t(weights=models.Swin_V2_T_Weights.DEFAULT if pretrained else None)
        elif variance == "swin_v2_s":
            self.model = models.swin_v2_s(weights=models.Swin_V2_S_Weights.DEFAULT if pretrained else None)
        elif variance == "swin_v2_b":
            self.model = models.swin_v2_b(weights=models.Swin_V2_B_Weights.DEFAULT if pretrained else None)

        del self.model.norm
        del self.model.permute
        del self.model.avgpool
        del self.model.flatten
        del self.model.head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.features[:4](x)
        feat1 = x.permute(0, 3, 1, 2)
        x = self.model.features[4:6](x)
        feat2 = x.permute(0, 3, 1, 2)
        x = self.model.features[6:](x)
        feat3 = x.permute(0, 3, 1, 2)

        return [feat1, feat2, feat3]

# |       | t    | s    | b    |
# | ----- | ---- | ---- | ---- |
# | feat1 | 192  | 192  | 256  |
# | feat2 | 384  | 384  | 512  |
# | feat3 | 768  | 768  | 1024 |
