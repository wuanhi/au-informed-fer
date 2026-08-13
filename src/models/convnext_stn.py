import torch.nn as nn

from src.models.convnext_backbone import build_convnext_tiny
from src.models.stn import SpatialTransformerNetwork


class STNConvNeXtTiny(nn.Module):
    """
    A1 architecture:

        Input
        -> STN
        -> pure ConvNeXt-Tiny
        -> 7-class logits
    """

    def __init__(
        self,
        num_classes=7,
        pretrained=True,
        drop_path_rate=0.1,
    ):
        super().__init__()

        # New module in A1
        self.stn = SpatialTransformerNetwork()

        # Same pure ConvNeXt-Tiny backbone as A0
        self.backbone = build_convnext_tiny(
            num_classes=num_classes,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x):
        # Input -> STN
        x = self.stn(x)

        # transformed x -> ConvNeXt
        logits = self.backbone(x)

        return logits

    def forward_stn_with_theta(self, x):
        """
        Diagnostic helper for post-training A1 analysis.
        """

        return self.stn.forward_with_theta(x)

    def get_sampling_grid(self, x):
        return self.stn.get_sampling_grid(x)


def build_stn_convnext_tiny(
    num_classes=7,
    pretrained=True,
    drop_path_rate=0.1,
):
    return STNConvNeXtTiny(
        num_classes=num_classes,
        pretrained=pretrained,
        drop_path_rate=drop_path_rate,
    )