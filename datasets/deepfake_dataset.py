"""
DeepfakeFrameDataset
====================
Scans  root/split/class_name/video_folder/image.*
Each image is one independent sample (frame-level).

Label mapping
-------------
  original                                    -> 0  (real)
  Deepfakes / Face2Face / FaceShifter /
  FaceSwap / NeuralTextures                   -> 1  (fake)
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

    Train: resize, flip, rotate, random-resized-crop (zoom), blur,
           gaussian noise, normalize, ToTensor.
    Val / Test: resize, normalize, ToTensor.

    Note on albumentations API compatibility
    ----------------------------------------
    RandomResizedCrop uses ``height`` / ``width`` kwargs (albumentations < 1.4).
    If you use albumentations >= 1.4, replace them with ``size=(image_size, image_size)``.
    """
    if split == "train":
        return A.Compose([
            # Base resize — ensures fixed size even when RandomResizedCrop is skipped
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            # Zoom in / zoom out: scale < 1 zooms in, scale > 1 zooms out
            A.RandomResizedCrop(
                height=image_size,
                width=image_size,
                scale=(0.7, 1.3),
                ratio=(0.75, 1.33),
                p=0.5,
            ),
            # Blur (randomly pick one type)
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
            ], p=0.3),
            # Gaussian noise
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        # val / test — deterministic
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
        └── split/                 # train | val | test
            ├── original/
            │   └── <video_id>/
            │       ├── 000.jpg
            │       └── ...
            └── Deepfakes/
                └── <video_id>/
                    ├── 000.jpg
                    └── ...

    Each image file is a single sample; no video-level aggregation.
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            root_dir: path to the dataset root folder
            split:    one of "train", "val", "test"
            transform: albumentations Compose (or any callable that accepts
                       ``image=np.ndarray`` keyword and returns a dict)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.samples: list[Tuple[str, int]] = []
        self._scan()

    def _scan(self) -> None:
        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}\n"
                f"Expected structure: {self.root_dir}/<split>/<class>/<video>/<image>"
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
                # Unknown class — skip silently
                continue

            for video_dir in sorted(class_dir.iterdir()):
                if not video_dir.is_dir():
                    continue
                for img_path in sorted(video_dir.iterdir()):
                    if img_path.suffix.lower() in SUPPORTED_EXTS:
                        self.samples.append((str(img_path), label))

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
            image  : FloatTensor [3, H, W]
            label  : int  (0 = real, 1 = fake)
            path   : str  (absolute path to the image file)
        """
        img_path, label = self.samples[idx]

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"cv2.imread failed for: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]   # FloatTensor after ToTensorV2

        return image, label, img_path
