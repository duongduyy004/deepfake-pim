"""
Perturbation-based Invariance Module (PIM)
==========================================
Reference: "Improving Generalization of Deepfake Detectors by
            Imposing Gradient Regularization" (2024)

PIM is inserted after stage 0 of ConvNeXt.
It computes channel-wise spatial statistics (mu, sigma), normalises the
feature map, then optionally re-scales it with perturbed statistics.

Usage during training
---------------------
  1. Forward pass with delta_mu=None, delta_sigma=None
     → PIM returns the feature unchanged (identity) but exposes mu / sigma
       for gradient computation.
  2. Compute CE_clean; derive delta_mu / delta_sigma from its gradients.
  3. Second forward pass with delta_mu / delta_sigma attached.
     → PIM returns the perturbed feature.

At inference: the model calls forward() with use_pim=False, bypassing PIM
entirely.
"""

import torch
import torch.nn as nn


class PIM(nn.Module):
    """Perturbation-based Invariance Module."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        feat: torch.Tensor,
        delta_mu: torch.Tensor | None = None,
        delta_sigma: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            feat:        [B, C, H, W]  feature map from stage 0
            delta_mu:    [B, C, 1, 1]  additive mean perturbation (or None)
            delta_sigma: [B, C, 1, 1]  additive std  perturbation (or None)

        Returns:
            feat_out: [B, C, H, W]  (perturbed or identity-reconstructed)
            mu:       [B, C, 1, 1]  channel-wise spatial mean
            sigma:    [B, C, 1, 1]  channel-wise spatial std
        """
        # Channel-wise statistics over spatial dims H, W
        mu = feat.mean(dim=[2, 3], keepdim=True)       # [B, C, 1, 1]
        sigma = feat.std(dim=[2, 3], keepdim=True)     # [B, C, 1, 1]

        # Instance-normalise
        feat_norm = (feat - mu) / (sigma + self.eps)   # [B, C, H, W]

        if delta_mu is not None and delta_sigma is not None:
            # Re-scale with perturbed statistics
            feat_out = feat_norm * (sigma + delta_sigma) + (mu + delta_mu)
        else:
            # Identity reconstruction — keeps mu/sigma in the graph
            feat_out = feat_norm * sigma + mu

        return feat_out, mu, sigma
