"""
PIM gradient-regularization loss
=================================
Reference: "Improving Generalization of Deepfake Detectors by
            Imposing Gradient Regularization" (2024)

Algorithm
---------
1. Forward pass 1  (use_pim=True, no perturbation)
   → logits_clean, mu, sigma
2. CE_clean = CrossEntropy(logits_clean, labels)
3. grad_mu, grad_sigma = ∂CE_clean / ∂mu, ∂CE_clean / ∂sigma
4. grad_norm = sqrt( ||grad_mu||² + ||grad_sigma||² ) + eps
5. delta_mu   = r * grad_mu   / grad_norm
   delta_sigma = r * grad_sigma / grad_norm
6. Forward pass 2  (use_pim=True, with delta_mu/sigma detached)
   → logits_perturbed
7. CE_perturbed = CrossEntropy(logits_perturbed, labels)
8. loss = (1 - alpha) * CE_clean + alpha * CE_perturbed

With alpha=1.0 the training signal comes entirely from CE_perturbed,
while CE_clean only provides the gradient direction for the perturbation.
"""

import torch
import torch.nn as nn


def pim_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    r: float,
    criterion: nn.Module,
) -> tuple[torch.Tensor, float, float, torch.Tensor]:
    """
    Compute the PIM gradient-regularization loss.

    Args:
        model:     ConvNeXtPIMDetector (must be in train mode)
        images:    [B, 3, H, W]  on device
        labels:    [B]           on device
        alpha:     weight for CE_perturbed in the final loss (1.0 → only perturbed)
        r:         perturbation magnitude
        criterion: nn.CrossEntropyLoss (reduction='mean')

    Returns:
        loss:            combined loss (scalar, graph attached — ready for .backward())
        ce_clean_val:    CE_clean  as Python float (for logging)
        ce_perturbed_val:CE_perturbed as Python float (for logging)
        logits_clean:    detached [B, C] tensor (for training metrics)
    """

    # ------------------------------------------------------------------
    # Pass 1: clean forward — obtain mu / sigma for gradient computation
    # ------------------------------------------------------------------
    logits_clean, mu, sigma = model(images, use_pim=True, delta_mu=None, delta_sigma=None)
    ce_clean = criterion(logits_clean, labels)

    # ------------------------------------------------------------------
    # Gradient of CE_clean w.r.t. mu and sigma
    # retain_graph=True: keeps the graph alive so loss.backward() can
    # later back-prop through the (1-alpha)*ce_clean term.
    # ------------------------------------------------------------------
    grad_mu, grad_sigma = torch.autograd.grad(
        ce_clean,
        [mu, sigma],
        retain_graph=True,   # graph still needed for loss.backward()
        create_graph=False,  # we do NOT differentiate through the gradient
    )

    # ------------------------------------------------------------------
    # Normalised perturbation  (global norm over all B, C elements)
    # ------------------------------------------------------------------
    grad_norm = torch.sqrt(
        (grad_mu ** 2).sum() + (grad_sigma ** 2).sum()
    ) + 1e-8

    delta_mu = r * grad_mu / grad_norm        # [B, C, 1, 1]
    delta_sigma = r * grad_sigma / grad_norm  # [B, C, 1, 1]

    # ------------------------------------------------------------------
    # Pass 2: perturbed forward
    # Detach perturbations so gradients flow only through model params,
    # not back into the first pass's graph.
    # ------------------------------------------------------------------
    logits_perturbed, _, _ = model(
        images,
        use_pim=True,
        delta_mu=delta_mu.detach(),
        delta_sigma=delta_sigma.detach(),
    )
    ce_perturbed = criterion(logits_perturbed, labels)

    # ------------------------------------------------------------------
    # Combined loss
    # ------------------------------------------------------------------
    loss = (1.0 - alpha) * ce_clean + alpha * ce_perturbed

    return loss, ce_clean.item(), ce_perturbed.item(), logits_clean.detach()
