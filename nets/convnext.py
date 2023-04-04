import torch
from torch import nn
from torchvision import models


class ConvNeXt(nn.Module):
    def __init__(self, variance: str = "convnext_tiny", pretrained=True) -> None:
        """convnext backbone

        Args:
            variance (str, optional): convnext version in [convnext_tiny, convnext_small, convnext_base, convnext_large]. Defaults to "convnext_tiny".
        """
        super().__init__()
        if variance == "convnext_tiny":
            self.model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        elif variance == "convnext_small":
            self.model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None)
        elif variance == "convnext_base":
            self.model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None)
        elif variance == "convnext_large":
            self.model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.DEFAULT if pretrained else None)

        del self.model.avgpool
        del self.model.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.features[:4](x)
        feat1 = x
        x = self.model.features[4:6](x)
        feat2 = x
        x = self.model.features[6:](x)
        feat3 = x

        return [feat1, feat2, feat3]


# | ch    | t    | s    | b    | l    |
# | ----- | ---- | ---- | ---- | ---- |
# | feat1 | 192  | 192  | 256  | 384  |
# | feat2 | 384  | 384  | 512  | 768  |
# | feat3 | 768  | 768  | 1024 | 1536 |
