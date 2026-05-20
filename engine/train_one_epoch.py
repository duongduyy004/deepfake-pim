"""
Training loop for one epoch.

Uses PIM gradient-regularization loss (pim_loss).
Metrics are computed on clean logits (use_pim=False equivalent) so they
reflect the model's inference behaviour.
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from losses.pim_loss import pim_loss
from utils.metrics import compute_metrics


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha: float,
    r: float,
) -> dict:
    """
    Train the model for one epoch.

    Args:
        model:      ConvNeXtPIMDetector
        dataloader: training DataLoader
        optimizer:  AdamW (or any optimizer)
        device:     torch.device
        alpha:      PIM loss weight for CE_perturbed
        r:          PIM perturbation magnitude

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

        # Compute PIM loss; logits_clean is detached and used only for metrics
        loss, ce_clean, ce_perturbed, logits_clean = pim_loss(
            model, images, labels, alpha=alpha, r=r, criterion=criterion
        )

        loss.backward()
        optimizer.step()

        # Collect predictions (from clean logits = inference behaviour)
        with torch.no_grad():
            probs = torch.softmax(logits_clean, dim=1)[:, 1]  # P(fake)
            preds = logits_clean.argmax(dim=1)

        total_loss += loss.item()
        y_true_all.extend(labels.cpu().numpy())
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
