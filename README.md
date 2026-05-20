# convnext-pim-detector

PyTorch implementation of a deepfake detector based on:

> **"Improving Generalization of Deepfake Detectors by Imposing Gradient Regularization"** (2024)

Backbone: **ConvNeXt-Base** · Regularization: **PIM (Perturbation-based Invariance Module)**

---

## How it works

PIM is inserted after stage 0 of ConvNeXt. During training only, it computes channel-wise spatial statistics (μ, σ) of the feature map, then perturbs them in the direction of the gradient of the clean loss. This forces the model to be invariant to style-level perturbations and improves cross-dataset generalization.

```
Input
  └─ stem → stage 0 → [PIM] → stage 1-3 → head → logits
                        ↑
                 training only
```

**Training loss (two forward passes):**

```
Pass 1  →  CE_clean  →  ∂CE_clean/∂μ, ∂CE_clean/∂σ
                              ↓
                   δμ = r · grad_μ / ‖grad‖
                   δσ = r · grad_σ / ‖grad‖
Pass 2  →  CE_perturbed  (with δμ, δσ applied)

L = (1 − α) · CE_clean + α · CE_perturbed
```

Default: `α = 1.0`, `r = 0.1` → loss is entirely CE_perturbed.

---

## Project structure

```
convnext-pim-detector/
├── configs/
│   └── config.yaml          # all hyperparameters
├── datasets/
│   └── deepfake_dataset.py  # DeepfakeFrameDataset + get_transforms
├── models/
│   ├── pim.py               # Perturbation-based Invariance Module
│   └── convnext_pim.py      # ConvNeXt-Base with PIM injected
├── losses/
│   └── pim_loss.py          # two-pass gradient regularization loss
├── engine/
│   ├── train_one_epoch.py   # training loop
│   ├── valid_on_test.py     # eval loop (PIM disabled)
│   └── trainer.py           # full loop, early stopping, checkpointing
├── utils/
│   ├── metrics.py           # Acc / F1 / Precision / Recall / AUC
│   ├── seed.py              # global seed
│   └── checkpoint.py        # save / load checkpoint
├── train.py                 # entry point
└── requirements.txt
```

---

## Dataset structure

```
root/
├── train/
│   ├── original/
│   │   └── <video_id>/  ← 64 cropped face frames
│   ├── Deepfakes/
│   │   └── <video_id>/
│   ├── Face2Face/
│   ├── FaceShifter/
│   ├── FaceSwap/
│   └── NeuralTextures/
├── val/
│   └── (same layout)
└── test/
    └── (same layout)
```

- Each image file is one independent sample (frame-level training).
- **Label:** `original` → 0 (real) · all other classes → 1 (fake)
- Supported formats: `.jpg` `.jpeg` `.png` `.bmp` `.webp`

---

## Installation

```bash
pip install -r requirements.txt
```

> **albumentations note:** `RandomResizedCrop` uses `height=`/`width=` kwargs (albumentations < 1.4).  
> For albumentations ≥ 1.4, change line ~39 in `datasets/deepfake_dataset.py` to `size=(224, 224)`.

---

## Training

**Minimal:**
```bash
python train.py --root_dir "path/to/dataset" --save_dir "checkpoints"
```

**Full options:**
```bash
python train.py \
    --root_dir  "F:/DeepFakedata/face_crops" \
    --save_dir  "checkpoints" \
    --config    "configs/config.yaml" \
    --batch_size 32 \
    --num_workers 4 \
    --epochs    30 \
    --lr        0.0001
```

CLI arguments override the corresponding values in `config.yaml`.

---

## Configuration

`configs/config.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `root_dir` | `data/face_crops` | Dataset root folder |
| `save_dir` | `checkpoints` | Checkpoint output folder |
| `image_size` | `224` | Input resolution |
| `batch_size` | `32` | |
| `num_workers` | `4` | DataLoader workers |
| `epochs` | `30` | Max training epochs |
| `lr` | `0.0001` | AdamW learning rate |
| `weight_decay` | `0.0001` | AdamW weight decay |
| `alpha` | `1.0` | PIM loss weight (1.0 = only perturbed loss) |
| `r` | `0.1` | PIM perturbation magnitude |
| `patience_scheduler` | `2` | ReduceLROnPlateau patience |
| `patience_early_stopping` | `7` | Early stopping patience |
| `num_classes` | `2` | Output classes (real / fake) |
| `model_name` | `convnext_base` | timm model identifier |
| `pretrained` | `true` | Use ImageNet pretrained weights |
| `seed` | `42` | Global random seed |

---

## Checkpoints

The best checkpoint (lowest `val_loss`) is saved to `<save_dir>/best_model.pth` and contains:

```python
{
    "epoch":                int,
    "model_state_dict":     ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "best_val_loss":        float,
    "config":               dict,
}
```

---

## Output metrics

Reported after every epoch (train + val) and once on test after training:

```
Loss | Acc | F1 | Precision | Recall | AUC
```
