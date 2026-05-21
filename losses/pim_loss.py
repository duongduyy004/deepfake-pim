"""
PIM gradient-regularization loss (with optional MixUp support)
===============================================================
Reference: "Improving Generalization of Deepfake Detectors by
            Imposing Gradient Regularization" (2024)

Algorithm
---------
1. Forward pass 1  (use_pim=True, no perturbation)
   -> logits_clean, mu, sigma
2. CE_clean = CrossEntropy(logits_clean, labels)  [or MixUp variant]
3. grad_mu, grad_sigma = dCE_clean / d(mu, sigma)
4. grad_norm = sqrt( ||grad_mu||^2 + ||grad_sigma||^2 ) + eps
5. delta_mu   = r * grad_mu   / grad_norm
   delta_sigma = r * grad_sigma / grad_norm
6. Forward pass 2  (use_pim=True, with delta_mu/sigma detached)
   -> logits_perturbed
7. CE_perturbed = CrossEntropy(logits_perturbed, labels)  [or MixUp variant]
8. loss = (1 - alpha) * CE_clean + alpha * CE_perturbed

MixUp: when labels_b and lam are provided, CE becomes
       lam * CE(logits, labels_a) + (1 - lam) * CE(logits, labels_b)
"""

import torch
import torch.nn as nn


def _ce_loss(
    criterion: nn.Module,
    logits: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor | None,
    lam: float,
) -> torch.Tensor:
    if labels_b is not None and lam < 1.0:
        return lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
    return criterion(logits, labels_a)


def pim_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    r: float,
    criterion: nn.Module,
    labels_b: torch.Tensor | None = None,
    lam: float = 1.0,
) -> tuple[torch.Tensor, float, float, torch.Tensor]:
    """
    Compute the PIM gradient-regularization loss.

    Args:
        model:     EfficientNetB4PIMDetector (must be in train mode)
        images:    [B, 3, H, W]  on device  (may be MixUp-mixed)
        labels:    [B]           on device  (labels_a when MixUp is used)
        alpha:     weight for CE_perturbed in the final loss
        r:         perturbation magnitude
        criterion: nn.CrossEntropyLoss (reduction='mean')
        labels_b:  [B] second labels for MixUp (None -> no MixUp)
        lam:       MixUp interpolation factor (1.0 -> no MixUp)

    Returns:
        loss:             combined loss (scalar, graph attached)
        ce_clean_val:     CE_clean  as Python float (for logging)
        ce_perturbed_val: CE_perturbed as Python float (for logging)
        logits_clean:     detached [B, C] tensor (for training metrics)
    """

    # ------------------------------------------------------------------
    # Pass 1: clean forward — obtain mu / sigma for gradient computation
    # ------------------------------------------------------------------
    logits_clean, mu, sigma = model(images, use_pim=True, delta_mu=None, delta_sigma=None)
    ce_clean = _ce_loss(criterion, logits_clean, labels, labels_b, lam)

    # ------------------------------------------------------------------
    # Gradient of CE_clean w.r.t. mu and sigma
    # ------------------------------------------------------------------
    grad_mu, grad_sigma = torch.autograd.grad(
        ce_clean,
        [mu, sigma],
        retain_graph=True,
        create_graph=False,
    )

    # ------------------------------------------------------------------
    # Normalised perturbation
    # ------------------------------------------------------------------
    grad_norm = torch.sqrt(
        (grad_mu ** 2).sum() + (grad_sigma ** 2).sum()
    ) + 1e-8

    delta_mu = r * grad_mu / grad_norm
    delta_sigma = r * grad_sigma / grad_norm

    # ------------------------------------------------------------------
    # Pass 2: perturbed forward
    # ------------------------------------------------------------------
    logits_perturbed, _, _ = model(
        images,
        use_pim=True,
        delta_mu=delta_mu.detach(),
        delta_sigma=delta_sigma.detach(),
    )
    ce_perturbed = _ce_loss(criterion, logits_perturbed, labels, labels_b, lam)

    # ------------------------------------------------------------------
    # Combined loss
    # ------------------------------------------------------------------
    loss = (1.0 - alpha) * ce_clean + alpha * ce_perturbed

    return loss, ce_clean.item(), ce_perturbed.item(), logits_clean.detach()
