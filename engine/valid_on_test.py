"""
Validation / test evaluation loop.

PIM is always disabled here (use_pim=False), matching real inference.
Shared by both val and test splits.
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.metrics import compute_metrics


def valid_on_test(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> dict:
    """
    Evaluate model on val or test split.

    Args:
        model:      ConvNeXtPIMDetector
        dataloader: val or test DataLoader
        device:     torch.device
        criterion:  nn.CrossEntropyLoss

    Returns:
        dict with keys: loss, acc, f1, precision, recall, auc
    """
    model.eval()

    total_loss = 0.0
    y_true_all: list[int] = []
    y_prob_all: list[float] = []
    y_pred_all: list[int] = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Eval", leave=False, dynamic_ncols=True)

        for images, labels, _ in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # PIM disabled — inference mode
            logits, _, _ = model(images, use_pim=False)
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)[:, 1]  # P(fake)
            preds = logits.argmax(dim=1)

            total_loss += loss.item()
            y_true_all.extend(labels.cpu().numpy())
            y_prob_all.extend(probs.cpu().numpy())
            y_pred_all.extend(preds.cpu().numpy())

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(
        np.array(y_true_all),
        np.array(y_prob_all),
        np.array(y_pred_all),
    )
    metrics["loss"] = avg_loss

    return metrics
