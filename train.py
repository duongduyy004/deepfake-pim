"""
Entry point for training ConvNeXt-Base + PIM deepfake detector.

Example
-------
  python train.py \\
      --root_dir "F:/DeepFakedata/face_crops" \\
      --save_dir "checkpoints" \\
      --batch_size 32 \\
      --epochs 30

CLI arguments override the values in configs/config.yaml.
"""

import argparse
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from datasets.deepfake_dataset import DeepfakeFrameDataset, get_transforms
from engine.trainer import train
from models.convnext_pim import ConvNeXtPIMDetector
from utils.seed import set_seed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ConvNeXt-Base + PIM deepfake detector"
    )
    parser.add_argument("--root_dir", type=str, default=None,
                        help="Path to dataset root folder (overrides config)")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory for saving checkpoints (overrides config)")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # CLI overrides
    overrides = {
        "root_dir": args.root_dir,
        "save_dir": args.save_dir,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "epochs": args.epochs,
        "lr": args.lr,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    # Reproducibility
    set_seed(config["seed"])

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Save directory
    Path(config["save_dir"]).mkdir(parents=True, exist_ok=True)

    # ---- Transforms ----
    train_tf = get_transforms("train", image_size=config["image_size"])
    val_tf = get_transforms("val",   image_size=config["image_size"])
    test_tf = get_transforms("test",  image_size=config["image_size"])

    # ---- Datasets ----
    train_ds = DeepfakeFrameDataset(config["root_dir"], "train", train_tf)
    val_ds   = DeepfakeFrameDataset(config["root_dir"], "val",   val_tf)
    test_ds  = DeepfakeFrameDataset(config["root_dir"], "test",  test_tf)

    print(f"\nDataset sizes  —  train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    # ---- DataLoaders ----
    loader_kwargs = dict(
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    # ---- Model ----
    print(f"\nBuilding {config['model_name']} + PIM  (pretrained={config['pretrained']}) …")
    model = ConvNeXtPIMDetector(
        model_name=config["model_name"],
        pretrained=config["pretrained"],
        num_classes=config["num_classes"],
    ).to(device)

    # ---- Optimizer ----
    optimizer = AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    # ---- Scheduler — steps on val_loss after every epoch ----
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=config["patience_scheduler"],
    )

    # ---- Train ----
    print(f"\nStarting training  (alpha={config['alpha']}, r={config['r']}) …\n")
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        save_dir=config["save_dir"],
    )


if __name__ == "__main__":
    main()
