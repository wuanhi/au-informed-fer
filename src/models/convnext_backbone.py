import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


def build_convnext_tiny(num_classes=7, pretrained=True):
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None

    model = convnext_tiny(weights=weights)

    # ConvNeXt-Tiny classifier:
    # [LayerNorm2d/LayerNorm, Flatten, Linear]
    in_features = model.classifier[2].in_features

    model.classifier[2] = nn.Linear(
        in_features=in_features,
        out_features=num_classes
    )

    return model