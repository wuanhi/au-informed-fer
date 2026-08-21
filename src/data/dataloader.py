# =============================================================================
# fer-stage1/src/data/dataloader.py
#
# Stage 1 – Ablation A2: Updated DataLoader factory
#
# Changes vs. A0/A1 baseline:
#   - Two new optional arguments on `build_dataloaders`:
#       `spatial_prior_generator` — instance of SpatialPriorGenerator (or None)
#       `landmarks_dict`          — loaded fer2013_landmarks.pkl dict (or None)
#   - Both are threaded into all three FER2013Dataset instances (train/val/test).
#   - When both are None (default), the function behaves identically to A0/A1
#     — the call site in scripts/train.py needs zero changes for baseline runs.
#   - All existing DataLoader kwargs (batch_size, num_workers, shuffle,
#     pin_memory) are preserved exactly as in the A0/A1 baseline.
# =============================================================================

from __future__ import annotations

from typing import Optional

from torch.utils.data import DataLoader

from src.data.fer2013_dataset import FER2013Dataset
from src.data.transforms import (
    get_train_transform,
    get_val_transform,
    get_test_transform,
)


def build_dataloaders(
    data_cfg,
    experiment_cfg,
    spatial_prior_generator=None,
    landmarks_dict: Optional[dict] = None,
):
    """
    Build and return (train_loader, val_loader, test_loader).

    Args:
        data_cfg (dict):
            Data configuration loaded from configs/data/fer2013.yaml.
            Expected keys: dataset.csv_path, splits.{train,val,test},
            image.size, dataloader.{train_batch_size, val_batch_size,
            test_batch_size, num_workers}.
        experiment_cfg (dict):
            Experiment configuration (e.g., A1 YAML).
            Expected key: augmentation.{resize_before_crop, rotation_degrees}.
        spatial_prior_generator (SpatialPriorGenerator, optional):
            Initialised SpatialPriorGenerator instance.
            Pass alongside `landmarks_dict` to enable A2 prior mode.
            When None (default), all datasets run in baseline mode and
            __getitem__ returns zero-filled dummy heatmaps.
        landmarks_dict (dict[str, np.ndarray | None], optional):
            The loaded fer2013_landmarks.pkl dictionary.
            Keys: f"{Usage}/{original_csv_row_index}" (e.g. "Training/0").
            Values: (68, 2) float32 numpy arrays or None.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    csv_path = data_cfg["dataset"]["csv_path"]

    image_size = data_cfg["image"]["size"]

    augmentation_cfg = experiment_cfg["augmentation"]

    # ------------------------------------------------------------------
    # Transforms — unchanged from A0/A1 baseline
    # ------------------------------------------------------------------
    train_transform = get_train_transform(
        image_size,
        augmentation_cfg,
    )

    val_transform = get_val_transform(
        image_size,
        augmentation_cfg,
    )

    test_transform = get_test_transform(
        image_size,
        augmentation_cfg,
    )

    # ------------------------------------------------------------------
    # Dataset instances — prior args threaded into all three splits
    # ------------------------------------------------------------------
    train_dataset = FER2013Dataset(
        csv_path=csv_path,
        split=data_cfg["splits"]["train"],
        transform=train_transform,
        spatial_prior_generator=spatial_prior_generator,
        landmarks_dict=landmarks_dict,
    )

    val_dataset = FER2013Dataset(
        csv_path=csv_path,
        split=data_cfg["splits"]["val"],
        transform=val_transform,
        spatial_prior_generator=spatial_prior_generator,
        landmarks_dict=landmarks_dict,
    )

    test_dataset = FER2013Dataset(
        csv_path=csv_path,
        split=data_cfg["splits"]["test"],
        transform=test_transform,
        spatial_prior_generator=spatial_prior_generator,
        landmarks_dict=landmarks_dict,
    )

    # ------------------------------------------------------------------
    # DataLoaders — all kwargs preserved exactly from A0/A1 baseline
    # ------------------------------------------------------------------
    loader_cfg = data_cfg["dataloader"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=loader_cfg["train_batch_size"],
        shuffle=True,
        num_workers=loader_cfg["num_workers"],
    )

    # EmoNeXt source:
    # validation batch_size = 1
    val_loader = DataLoader(
        val_dataset,
        batch_size=loader_cfg["val_batch_size"],
        shuffle=False,
    )

    # EmoNeXt source:
    # test batch_size = 32
    test_loader = DataLoader(
        test_dataset,
        batch_size=loader_cfg["test_batch_size"],
        shuffle=False,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )