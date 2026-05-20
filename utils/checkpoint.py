from pathlib import Path
import torch


def save_checkpoint(state: dict, save_path: str) -> None:
    """Save training checkpoint to disk."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, save_path)


def load_checkpoint(
    model,
    optimizer,
    scheduler,
    save_path: str,
    device: torch.device,
):
    """
    Load model + optimizer + scheduler state from checkpoint.

    Returns:
        epoch: int — epoch at which checkpoint was saved
        best_val_loss: float
    """
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"], checkpoint["best_val_loss"]
