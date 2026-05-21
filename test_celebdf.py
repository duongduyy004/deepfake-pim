"""
CelebDF evaluation script
==========================
Loads the best checkpoint (best_auc_model.pth) and evaluates on CelebDF.

Expected dataset structure::

    D:/duong_huy_ct7/deepfake-data/celeb-df/test/
    ├── fake/    # label = 1
    └── real/    # label = 0

Each folder contains frame images (.jpg, .png, etc.).

Usage
-----
  python test_celebdf.py
  python test_celebdf.py --ckpt checkpoints/best_auc_model.pth \\
                         --data_dir "D:/duong_huy_ct7/deepfake-data/celeb-df/test" \\
                         --config configs/config.yaml
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.efficientnet_pim import EfficientNetB4PIMDetector

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CELEBDF_LABEL_MAP = {"real": 0, "fake": 1}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CelebDFDataset(Dataset):
    """
    CelebDF test dataset.

    Directory layout::

        root_dir/
        ├── fake/
        │   ├── 000.jpg
        │   └── ...
        └── real/
            ├── 000.jpg
            └── ...
    """

    def __init__(self, root_dir: str, image_size: int = 224) -> None:
        self.root_dir = Path(root_dir)
        self.transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
        self.samples: list[tuple[str, int]] = []
        self._scan()

    def _scan(self) -> None:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"CelebDF test directory not found: {self.root_dir}")

        for class_name, label in CELEBDF_LABEL_MAP.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Class folder not found: {class_dir}\n"
                    f"Expected subfolders: fake/, real/"
                )
            img_paths = sorted([
                p for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
            ])
            for p in img_paths:
                self.samples.append((str(p), label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"cv2.imread failed for: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=image)
        return augmented["image"], label


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> dict:
    model.eval()

    y_true_all: list[int] = []
    y_prob_all: list[float] = []
    y_pred_all: list[int] = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="CelebDF Eval", dynamic_ncols=True)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            logits, _, _ = model(images, use_pim=False)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)

            y_true_all.extend(labels.cpu().numpy())
            y_prob_all.extend(probs.cpu().numpy())
            y_pred_all.extend(preds.cpu().numpy())

    y_true = np.array(y_true_all)
    y_prob = np.array(y_prob_all)
    y_pred = np.array(y_pred_all)

    acc       = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0

    return {
        "accuracy":  acc,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "auc":       auc,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate on CelebDF test set")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/best_auc_model.pth",
        help="Path to best_auc_model.pth",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="D:/duong_huy_ct7/deepfake-data/celeb-df/test",
        help="Path to CelebDF test directory (contains fake/ and real/ subfolders)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
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

    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Dataset ----
    print(f"\nLoading CelebDF test set from: {args.data_dir}")
    dataset = CelebDFDataset(
        root_dir=args.data_dir,
        image_size=config.get("image_size", 224),
    )
    real_count = sum(1 for _, lbl in dataset.samples if lbl == 0)
    fake_count = sum(1 for _, lbl in dataset.samples if lbl == 1)
    print(f"  Total samples: {len(dataset)}  (real={real_count}, fake={fake_count})")

    dataloader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 32),
        num_workers=config.get("num_workers", 4),
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    # ---- Model ----
    print(f"\nBuilding model ...")
    model = EfficientNetB4PIMDetector(
        pretrained=False,
        num_classes=config.get("num_classes", 2),
        dropout_rate=config.get("dropout_rate", 0.5),
    ).to(device)

    # ---- Load checkpoint ----
    ckpt_path = args.ckpt
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    saved_epoch = checkpoint.get("epoch", "?")
    saved_auc   = checkpoint.get("best_val_auc", "?")
    print(f"  Checkpoint from epoch {saved_epoch}, best_val_auc={saved_auc}")

    # ---- Evaluate ----
    print("\nRunning evaluation on CelebDF ...")
    metrics = evaluate(model, dataloader, device)

    # ---- Print results ----
    sep = "=" * 45
    print(f"\n{sep}")
    print("  CelebDF Evaluation Results")
    print(sep)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-score  : {metrics['f1']:.4f}")
    print(f"  AUC       : {metrics['auc']:.4f}")
    print(sep)


if __name__ == "__main__":
    main()
