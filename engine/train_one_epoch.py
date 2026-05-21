"""
Training loop for one epoch.

Uses PIM gradient-regularization loss with optional MixUp augmentation.
Metrics are computed on clean logits (use_pim=False equivalent) so they
reflect the model's inference behaviour.
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from losses.pim_loss import pim_loss
from utils.metrics import compute_metrics


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply MixUp to a batch.

    Args:
        x:     [B, C, H, W] input images
        y:     [B] integer labels
        alpha: Beta distribution parameter (0 -> disabled)

    Returns:
        mixed_x:  [B, C, H, W]
        labels_a: [B]  (original labels)
        labels_b: [B]  (shuffled labels)
        lam:      float interpolation factor
    """
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0

    batch_size = x.size(0)
    idx = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha: float,
    r: float,
    mixup_alpha: float = 0.0,
) -> dict:
    """
    Train the model for one epoch.

    Args:
        model:       EfficientNetB4PIMDetector
        dataloader:  training DataLoader
        optimizer:   AdamW (or any optimizer)
        device:      torch.device
        alpha:       PIM loss weight for CE_perturbed
        r:           PIM perturbation magnitude
        mixup_alpha: MixUp Beta parameter (0 = disabled)

    Returns:
        dict with keys: loss, acc, f1, precision, recall, auc
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    y_true_all: list[int] = []
    y_prob_all: list[float] = []
    y_pred_all: list[int] = []

    pbar = tqdm(dataloader, desc="Train", leave=False, dynamic_ncols=True)

    for images, labels, _ in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # MixUp augmentation (only when mixup_alpha > 0)
        if mixup_alpha > 0:
            images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
        else:
            labels_a, labels_b, lam = labels, None, 1.0

        loss, ce_clean, ce_perturbed, logits_clean = pim_loss(
            model, images, labels_a,
            alpha=alpha, r=r, criterion=criterion,
            labels_b=labels_b, lam=lam,
        )

        loss.backward()
        optimizer.step()

        # Collect predictions from clean logits (inference behaviour)
        with torch.no_grad():
            probs = torch.softmax(logits_clean, dim=1)[:, 1]  # P(fake)
            preds = logits_clean.argmax(dim=1)

        total_loss += loss.item()
        # Use original labels (labels_a) for metrics
        y_true_all.extend(labels_a.cpu().numpy())
        y_prob_all.extend(probs.cpu().numpy())
        y_pred_all.extend(preds.cpu().numpy())

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            ce_c=f"{ce_clean:.4f}",
            ce_p=f"{ce_perturbed:.4f}",
        )

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(
        np.array(y_true_all),
        np.array(y_prob_all),
        np.array(y_pred_all),
    )
    metrics["loss"] = avg_loss

    return metrics
