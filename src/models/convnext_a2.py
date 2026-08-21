# =============================================================================
# fer-stage1/src/models/convnext_a2.py
#
# Stage 1 – Ablation A2: ConvNeXt-Tiny + AU-Informed Attention Branch
#
# Architecture (NO STN):
#   Input (B, 3, 224, 224)
#   → Stem  + Stage 0   (B,  96, 56, 56)
#   → DS1   + Stage 1   (B, 192, 28, 28)
#   → DS2   + Stage 2   (B, 384, 14, 14)  ← F
#   → Attention Branch  (B,   1, 14, 14)  ← A   [A2 addition]
#   → F_prime = F * A   (B, 384, 14, 14)         [A2 addition]
#   → DS3   + Stage 3   (B, 768,  7,  7)
#   → Global Avg Pool   (B, 768)
#   → LayerNorm         (B, 768)
#   → Head              (B, num_classes)
#
# Returns: (logits, A)
#   logits — (B, num_classes) classification output
#   A      — (B, 1, 14, 14)  predicted spatial attention map in [0, 1]
#             used by the A2 loss to align with the prior heatmap P.
#
# Stage indexing note (0-based):
#   backbone.downsample_layers[i] / backbone.stages[i]
#   i=0 → Stem  + Stage 0  (dims[0]= 96)
#   i=1 → DS1   + Stage 1  (dims[1]=192)
#   i=2 → DS2   + Stage 2  (dims[2]=384)  ← attention tap
#   i=3 → DS3   + Stage 3  (dims[3]=768)
# =============================================================================

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.convnext_backbone import build_convnext_tiny

# Channel dimension at the output of backbone Stage 2 (index 2, 0-based).
# ConvNeXt-Tiny dims = [96, 192, 384, 768]; Stage 2 output = dims[2].
_STAGE2_CHANNELS: int = 384


class ConvNeXt_A2(nn.Module):
    """
    A2 architecture (pure ConvNeXt-Tiny + Attention Branch, no STN):

        Input
        → Stages 0-1-2         (feature extraction up to 14×14)
        → Attention Branch     (predict spatial attention map A)
        → Feature Modulation   (F_prime = F * A)
        → Stage 3 + GAP + Head (classification)

    Args:
        num_classes    (int)  : Number of output classes (default 7).
        pretrained     (bool) : Load ImageNet-22K weights for the backbone.
        drop_path_rate (float): Stochastic depth rate passed to the backbone.

    Forward input:
        x (Tensor): (B, 3, 224, 224)

    Forward output:
        logits (Tensor): (B, num_classes)
        A      (Tensor): (B, 1, 14, 14) — spatial attention map in [0, 1]
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # ConvNeXt-Tiny backbone — loaded with pretrained weights.
        # We do NOT call backbone.forward() directly; instead we manually
        # iterate its downsample_layers and stages to tap the intermediate
        # tensor F at the output of Stage 2 (B, 384, 14, 14).
        # ------------------------------------------------------------------
        self.backbone = build_convnext_tiny(
            num_classes=num_classes,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
        )

        # ------------------------------------------------------------------
        # Attention Branch
        #
        # Input : F — (B, 384, 14, 14)  output of backbone Stage 2
        # Output: A — (B,   1, 14, 14)  spatial attention map in [0, 1]
        #
        # 1×1 Conv projects 384 channels → 1 channel (no spatial shrinkage).
        # Sigmoid squashes the output to [0, 1] so it acts as a soft mask.
        # ------------------------------------------------------------------
        self.attention_branch = nn.Sequential(
            nn.Conv2d(_STAGE2_CHANNELS, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # Initialise to near-zero weights so the mask starts near uniform
        # (σ(~0) ≈ 0.5) and does not disturb pretrained features early in
        # training.
        nn.init.trunc_normal_(self.attention_branch[0].weight, std=0.02)
        nn.init.zeros_(self.attention_branch[0].bias)

    # -----------------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, 224, 224) input image batch.

        Returns:
            logits: (B, num_classes)
            A:      (B, 1, 14, 14) spatial attention map from the branch.
        """
        # ------------------------------------------------------------------
        # Backbone Stages 0 → 2
        #
        # Each step:  x = downsample_layers[i](x)  then  x = stages[i](x)
        #
        # After i=0  (Stem  + Stage 0): (B,  96, 56, 56)
        # After i=1  (DS1   + Stage 1): (B, 192, 28, 28)
        # After i=2  (DS2   + Stage 2): (B, 384, 14, 14)  ← F
        # ------------------------------------------------------------------
        for i in range(3):
            x = self.backbone.downsample_layers[i](x)
            x = self.backbone.stages[i](x)

        F = x  # (B, 384, 14, 14)

        # ------------------------------------------------------------------
        # Attention Branch
        # A = σ( Conv1×1(F) )    shape: (B, 1, 14, 14)
        # ------------------------------------------------------------------
        A = self.attention_branch(F)  # (B, 1, 14, 14)

        # ------------------------------------------------------------------
        # Feature Modulation
        # F_prime = F * A
        # Broadcasting: (B, 384, 14, 14) * (B, 1, 14, 14) → (B, 384, 14, 14)
        # Each of the 384 feature channels is scaled by the same spatial mask.
        # ------------------------------------------------------------------
        F_prime = F * A  # (B, 384, 14, 14)

        # ------------------------------------------------------------------
        # Backbone Stage 3
        # DS3 + Stage 3: (B, 384, 14, 14) → (B, 768, 7, 7)
        # ------------------------------------------------------------------
        x = self.backbone.downsample_layers[3](F_prime)
        x = self.backbone.stages[3](x)  # (B, 768, 7, 7)

        # ------------------------------------------------------------------
        # Global Average Pooling + LayerNorm + Classification Head
        # Mirrors ConvNeXt.forward_features() → head() exactly.
        # ------------------------------------------------------------------
        x = x.mean([-2, -1])            # (B, 768) — GAP over H and W
        x = self.backbone.norm(x)       # (B, 768) — final LayerNorm
        logits = self.backbone.head(x)  # (B, num_classes)

        return logits, A


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_convnext_a2(
    num_classes: int = 7,
    pretrained: bool = True,
    drop_path_rate: float = 0.1,
) -> ConvNeXt_A2:
    """
    Instantiate and return a ConvNeXt_A2 model.

    Usage:
        from src.models.convnext_a2 import build_convnext_a2
        model = build_convnext_a2(num_classes=7, pretrained=True)
        logits, A = model(images)   # A is (B, 1, 14, 14)
    """
    return ConvNeXt_A2(
        num_classes=num_classes,
        pretrained=pretrained,
        drop_path_rate=drop_path_rate,
    )