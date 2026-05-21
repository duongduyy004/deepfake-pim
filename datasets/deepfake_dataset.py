"""
DeepfakeFrameDataset
====================
Scans  root/split/class_name/image.*
Images are stored directly inside each class folder (no video subfolder).

Label mapping
-------------
  original                                    -> 0  (real)
  Deepfakes / Face2Face / FaceShifter /
  FaceSwap / NeuralTextures                   -> 1  (fake)

Upsampling (train split only)
------------------------------
  original_upsample_factor = N  (total multiplier):
    - 1 base copy per image  -> base_transform  (no augmentation)
    - N-1 extra copies        -> train_transform (with augmentation)
  Fake images always use train_transform in the train split.

Val / Test
----------
  All images use the provided transform (resize + normalize only).
"""

from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FAKE_CLASSES = {"Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"}
REAL_CLASS = "original"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(split: str, image_size: int = 224) -> A.Compose:
    """
    Return albumentations transform pipeline for the requested split.

    Train: resize, flip, rotate, random-resized-crop, blur,
           gaussian noise, normalize, ToTensor.
    Val / Test: resize, normalize, ToTensor.
    """
    if split == "train":
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.7, 1.0),
                ratio=(0.75, 1.33),
                p=0.5,
            ),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
            ], p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DeepfakeFrameDataset(Dataset):
    """
    Frame-level deepfake detection dataset.

    Directory layout expected::

        root_dir/
        └── split/              # train | val | test
            ├── original/
            │   ├── 000.jpg
            │   └── ...
            └── Deepfakes/
                ├── 000.jpg
                └── ...

    Images are placed directly in the class folder (no video subfolder).

    Train split — per-sample transform rules:
      original (base copy):      transform      (no augmentation)
      original (upsampled N-1):  train_transform (augmentation)
      fake:                      train_transform (augmentation)

    Val / Test split:
      all images use transform   (no augmentation)
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        transform: Optional[Callable] = None,
        upsample_factor: int = 1,
        train_transform: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            root_dir:         path to dataset root folder
            split:            one of "train", "val", "test"
            transform:        base transform — used for val/test and for the
                              original base copy in train (resize + normalize)
            upsample_factor:  total multiplier for original class in train split.
                              1 = no upsampling; 2 = 2x total (1 base + 1 augmented)
            train_transform:  augmentation transform used for upsampled original
                              copies and for all fake images in train split
        """
        self.root_dir = Path(root_dir)
        self.split = split
        # Each entry: (img_path, label, callable_transform)
        self.samples: list[Tuple[str, int, Optional[Callable]]] = []
        self._scan(transform, upsample_factor, train_transform)

    def _scan(
        self,
        base_transform: Optional[Callable],
        upsample_factor: int,
        train_transform: Optional[Callable],
    ) -> None:
        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}\n"
                f"Expected: {self.root_dir}/<split>/<class>/<image>"
            )

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name

            if class_name == REAL_CLASS:
                label = 0
            elif class_name in FAKE_CLASSES:
                label = 1
            else:
                continue

            img_paths = sorted([
                p for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
            ])

            if self.split == "train":
                if label == 0:  # original / real
                    # Base copy: no augmentation
                    for p in img_paths:
                        self.samples.append((str(p), 0, base_transform))
                    # Extra upsampled copies: with augmentation
                    for _ in range(upsample_factor - 1):
                        for p in img_paths:
                            self.samples.append((str(p), 0, train_transform))
                else:  # fake
                    for p in img_paths:
                        self.samples.append((str(p), 1, train_transform))
            else:  # val / test
                for p in img_paths:
                    self.samples.append((str(p), label, base_transform))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in {split_dir}. "
                "Check that class folders match expected names and images have "
                "supported extensions (.jpg .jpeg .png .bmp .webp)."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Returns:
            image : FloatTensor [3, H, W]
            label : int  (0 = real, 1 = fake)
            path  : str  (absolute path to the image file)
        """
        img_path, label, transform = self.samples[idx]

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"cv2.imread failed for: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if transform is not None:
            augmented = transform(image=image)
            image = augmented["image"]

        return image, label, img_path
