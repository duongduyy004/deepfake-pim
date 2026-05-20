"""
ConvNeXt-Base + PIM detector
============================
Architecture split:
  stem  →  stage 0  →  [PIM]  →  stage 1-3  →  head (pool + fc)

PIM is active only during training (use_pim=True).
Inference always calls forward(x, use_pim=False).
"""

import torch
import torch.nn as nn
import timm

from models.pim import PIM


class ConvNeXtPIMDetector(nn.Module):
    """ConvNeXt-Base with PIM injected after stage 0."""

    def __init__(
        self,
        model_name: str = "convnext_base",
        pretrained: bool = True,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        # Load the full timm backbone
        backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )

        # ---- Split backbone at the stage-0 / stage-1 boundary ----
        # timm ConvNeXt: backbone.stem + backbone.stages (Sequential of 4 stages)
        self.stem = backbone.stem                   # patchify stem (4×4 stride-4 conv + LN)
        self.stage0 = backbone.stages[0]            # 3 blocks, no spatial downsampling

        # Remaining stages (each includes a 2× downsampler + blocks)
        n_stages = len(backbone.stages)
        self.stages_rest = nn.Sequential(
            *[backbone.stages[i] for i in range(1, n_stages)]
        )

        # Classification head: global avg-pool + LayerNorm + Linear
        self.head = backbone.head

        # PIM module — inserted between stage0 and stages_rest
        self.pim = PIM()

    # ------------------------------------------------------------------
    # Sub-forward helpers (exposed for use in pim_loss)
    # ------------------------------------------------------------------

    def forward_features_until_stage1(self, x: torch.Tensor) -> torch.Tensor:
        """stem → stage 0  →  [B, C, H/4, W/4]"""
        x = self.stem(x)
        x = self.stage0(x)
        return x

    def forward_features_after_stage1(self, x: torch.Tensor) -> torch.Tensor:
        """stages 1-3 → head  →  [B, num_classes]"""
        x = self.stages_rest(x)
        x = self.head(x)   # head handles global avg-pool + norm + fc internally
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
