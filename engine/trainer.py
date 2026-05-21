"""
Main trainer: full training loop with early stopping and final test evaluation.

Checkpoint layout
-----------------
  epoch              — epoch number (1-based)
  model_state_dict
  optimizer_state_dict
  scheduler_state_dict
  best_val_auc
  config             — full config dict for reproducibility
"""

import os
from pathlib import Path

import torch
import torch.nn as nn

from engine.train_one_epoch import train_one_epoch
from engine.valid_on_test import valid_on_test
from utils.checkpoint import save_checkpoint

_SEP = "=" * 65


def _fmt(metrics: dict) -> str:
    return (
        f"Loss: {metrics['loss']:.4f}  "
        f"Acc: {metrics['acc']:.4f}  "
        f"F1: {metrics['f1']:.4f}  "
        f"Prec: {metrics['precision']:.4f}  "
        f"Rec: {metrics['recall']:.4f}  "
        f"AUC: {metrics['auc']:.4f}"
    )


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,          # ReduceLROnPlateau(mode='max')
    config: dict,
    device: torch.device,
    save_dir: str,
) -> dict:
    """
    Full training loop.

    Trains for up to config['epochs'] epochs with:
      - PIM gradient-regularization on the training set
      - MixUp augmentation (if mixup_alpha > 0)
      - ReduceLROnPlateau(mode='max') on val AUC after every epoch
      - Best checkpoint saving when val AUC improves -> best_auc_model.pth
      - Early stopping after config['early_stopping_patience'] stagnant epochs

    After training, loads the best checkpoint and evaluates on test_loader.

    Returns:
        test_metrics dict
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    best_ckpt_path = os.path.join(save_dir, "best_auc_model.pth")

    criterion = nn.CrossEntropyLoss()

    max_epochs = config.get("epochs", 30)
    patience_es = config.get("early_stopping_patience", 5)
    alpha = config.get("alpha", 1.0)
    r = config.get("r", 0.1)
    mixup_alpha = config.get("mixup_alpha", 0.0)

    best_val_auc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        print(f"\n{_SEP}")
        print(f"  Epoch {epoch:>3d} / {max_epochs}")
        print(_SEP)

        # ---- Train ----
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            alpha=alpha, r=r, mixup_alpha=mixup_alpha,
        )

        # ---- Validate ----
        val_metrics = valid_on_test(model, val_loader, device, criterion)

        # ---- Scheduler step on val AUC (mode='max') ----
        val_auc = val_metrics["auc"]
        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]["lr"]

        # ---- Log ----
        print(f"  Train  |  {_fmt(train_metrics)}")
        print(f"  Val    |  {_fmt(val_metrics)}")
        print(f"  LR: {current_lr:.2e}  |  Best Val AUC: {best_val_auc:.4f}")

        # ---- Checkpoint: save when val AUC improves ----
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0

            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_auc": best_val_auc,
                    "config": config,
                },
                best_ckpt_path,
            )
            print(f"  -> Checkpoint saved  (val_auc={best_val_auc:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement  ({epochs_no_improve}/{patience_es})")

        # ---- Early stopping ----
        if epochs_no_improve >= patience_es:
            print(f"\n  Early stopping triggered at epoch {epoch}.")
            break

    # ---- Final test evaluation on best checkpoint ----
    print(f"\n{_SEP}")
    print("  Loading best checkpoint for test evaluation ...")
    print(_SEP)

    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = valid_on_test(model, test_loader, device, criterion)
    print(f"  Test   |  {_fmt(test_metrics)}")

    return test_metrics
