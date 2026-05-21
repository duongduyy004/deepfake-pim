"""
EfficientNet-B4 + PIM detector
================================
Architecture:
  stem (conv_stem + bn1 + act1)
    -> stage0 (blocks[0] + blocks[1])
    -> [PIM]
    -> stages_rest (blocks[2:] + conv_head + bn2 + act2)
    -> head (global_pool + dropout + fc)

PIM injection point: after blocks[1] (stride-4 spatial resolution, 32 channels).
This is analogous to ConvNeXt stage0 output (1/4 spatial resolution).

PIM is active only during training (use_pim=True).
Inference always calls forward(x, use_pim=False).
"""

import torch
import torch.nn as nn
import timm

from models.pim import PIM


class EfficientNetB4PIMDetector(nn.Module):
    """EfficientNet-B4 with PIM injected after the second block stage."""

    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 2,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()

        backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=num_classes,
        )

        # Stem: conv_stem -> bn1 -> act1
        # Output: [B, 48, H/2, W/2]
        self.stem = nn.Sequential(
            backbone.conv_stem,
            backbone.bn1,
            backbone.act1,
        )

        # stage0: blocks[0] (stride 1, 24ch) + blocks[1] (stride 2, 32ch)
        # Output: [B, 32, H/4, W/4]  e.g. 56x56 for 224 input
        blocks_list = list(backbone.blocks)
        self.stage0 = nn.Sequential(*blocks_list[:2])

        # Remaining stages + feature head
        # blocks[2..6] + conv_head (->1792ch) + bn2 + act2
        self.stages_rest = nn.Sequential(
            *blocks_list[2:],
            backbone.conv_head,
            backbone.bn2,
            backbone.act2,
        )

        # Classification head: global pool -> dropout -> linear
        self.global_pool = backbone.global_pool
        self.head_dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(backbone.num_features, num_classes)

        # PIM module — inserted between stage0 and stages_rest
        self.pim = PIM()

    # ------------------------------------------------------------------
    # Sub-forward helpers (exposed for pim_loss)
    # ------------------------------------------------------------------

    def forward_features_until_stage1(self, x: torch.Tensor) -> torch.Tensor:
        """stem -> stage0  -> [B, 32, H/4, W/4]"""
        x = self.stem(x)
        x = self.stage0(x)
        return x

    def forward_features_after_stage1(self, x: torch.Tensor) -> torch.Tensor:
        """stages_rest -> global_pool -> dropout -> fc -> [B, num_classes]"""
        x = self.stages_rest(x)
        x = self.global_pool(x)   # [B, num_features]
        x = self.head_dropout(x)
        x = self.fc(x)
        return x

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        use_pim: bool = False,
        delta_mu: torch.Tensor | None = None,
        delta_sigma: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Args:
            x:           [B, 3, H, W]
            use_pim:     True only during training (pim_loss calls)
            delta_mu:    [B, C, 1, 1] perturbation for mean  (2nd forward pass)
            delta_sigma: [B, C, 1, 1] perturbation for std   (2nd forward pass)

        Returns:
            logits: [B, num_classes]
            mu:     [B, C, 1, 1] or None
            sigma:  [B, C, 1, 1] or None
        """
        feat = self.forward_features_until_stage1(x)

        if use_pim:
            feat, mu, sigma = self.pim(feat, delta_mu, delta_sigma)
        else:
            mu, sigma = None, None

        logits = self.forward_features_after_stage1(feat)
        return logits, mu, sigma
