# =============================================================================
# fer-stage1/src/data/fer2013_dataset.py
#
# Stage 1 – Ablation A2: Updated FER2013Dataset
#
# Changes vs. A0/A1 baseline:
#   - Two new optional constructor args:
#       `spatial_prior_generator` — instance of SpatialPriorGenerator
#       `landmarks_dict`          — the loaded fer2013_landmarks.pkl dict
#   - __getitem__ now returns a 4-tuple:
#       (image, label, heatmap_P, valid_mask)
#   - PKL key is reconstructed as f"{split}/{original_csv_row_index}",
#     matching the key scheme used in scripts/extract_landmarks.py.
#   - When prior args are absent (baseline / inference mode), __getitem__
#     returns zero-filled dummy tensors so the return signature is always
#     consistent and downstream code never needs to branch on tuple length.
# =============================================================================

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


EMOTION_MAP = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

# Spatial size of the dummy heatmap returned in baseline / no-prior mode.
# Must match SpatialPriorGenerator.target_size (default 14).
_DEFAULT_HEATMAP_SIZE: int = 14


class FER2013Dataset(Dataset):
    """
    PyTorch Dataset for FER2013.

    In A2 (prior-enabled) mode, each item is a 4-tuple:
        (image, label, heatmap_P, valid_mask)

    In baseline mode (prior args omitted), heatmap_P is a zero tensor of
    shape (1, 14, 14) and valid_mask is 0 — keeping the return signature
    identical so training loops never need to branch on tuple length.

    Args:
        csv_path (str):
            Path to the FER2013 CSV file.
        split (str):
            One of "Training", "PublicTest", "PrivateTest".
        transform (callable, optional):
            Torchvision transform pipeline applied to the PIL image.
        spatial_prior_generator (SpatialPriorGenerator, optional):
            Initialised SpatialPriorGenerator instance. When provided,
            `landmarks_dict` must also be provided.
        landmarks_dict (dict[str, np.ndarray | None], optional):
            The loaded fer2013_landmarks.pkl dictionary.
            Keys follow the scheme: f"{Usage}/{original_csv_row_index}"
            e.g. "Training/0", "PublicTest/7178".
            Values are (68, 2) float32 numpy arrays or None.
    """

    def __init__(
        self,
        csv_path: str,
        split: str = "Training",
        transform=None,
        spatial_prior_generator=None,
        landmarks_dict: Optional[dict] = None,
    ) -> None:
        # ------------------------------------------------------------------
        # Load & filter CSV — keep the original index as a column so we can
        # reconstruct the PKL key exactly as the extractor wrote it.
        #
        # extract_landmarks.py iterates df.iterrows() on the FULL csv and
        # stores:  key = f"{row['Usage']}/{idx}"   (idx = original row index)
        #
        # After reset_index(drop=True) the local __getitem__ idx would be
        # wrong for the pkl lookup, so we save the original index first.
        # ------------------------------------------------------------------
        full_df = pd.read_csv(csv_path)

        split_df = full_df[full_df["Usage"] == split].copy()

        # Store the original CSV row index; reset positional index for iloc[]
        split_df["_original_idx"] = split_df.index
        split_df = split_df.reset_index(drop=True)

        self.df = split_df
        self.split = split
        self.transform = transform

        # ------------------------------------------------------------------
        # Prior-mode setup
        # ------------------------------------------------------------------
        self.spatial_prior_generator = spatial_prior_generator
        self.landmarks_dict = landmarks_dict

        _prior_enabled = (
            self.spatial_prior_generator is not None
            and self.landmarks_dict is not None
        )

        if spatial_prior_generator is not None and landmarks_dict is None:
            logger.warning(
                "[FER2013Dataset] `spatial_prior_generator` was provided but "
                "`landmarks_dict` is None. Prior generation will be DISABLED. "
                "Pass both arguments together to enable A2 mode."
            )

        if landmarks_dict is not None and spatial_prior_generator is None:
            logger.warning(
                "[FER2013Dataset] `landmarks_dict` was provided but "
                "`spatial_prior_generator` is None. Prior generation will be DISABLED. "
                "Pass both arguments together to enable A2 mode."
            )

        logger.info(
            "[FER2013Dataset] split='%s'  rows=%d  prior_mode=%s",
            split,
            len(self.df),
            "ENABLED" if _prior_enabled else "DISABLED (baseline)",
        )

    # -----------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    # -----------------------------------------------------------------------

    def __getitem__(self, idx: int):
        """
        Returns:
            image      (Tensor)         : Transformed image tensor.
            label      (int)            : Emotion class index [0–6].
            heatmap_P  (Tensor [1,H,W]) : Semantic prior heatmap, or zeros.
            valid_mask (Tensor [1])     : 1 if heatmap is valid, else 0.
        """
        row = self.df.iloc[idx]

        # ------------------------------------------------------------------
        # Decode image (FER2013 stores pixels as a space-separated string)
        # ------------------------------------------------------------------
        pixels = np.fromstring(row["pixels"], dtype=np.uint8, sep=" ")
        image = pixels.reshape(48, 48)
        image = Image.fromarray(image, mode="L")
        label = int(row["emotion"])

        if self.transform:
            image = self.transform(image)

        # ------------------------------------------------------------------
        # Spatial prior — only when both prior args are supplied
        # ------------------------------------------------------------------
        if (
            self.spatial_prior_generator is not None
            and self.landmarks_dict is not None
        ):
            # Reconstruct the PKL key using the ORIGINAL CSV row index.
            # This matches the key written by extract_landmarks.py:
            #   key = f"{row['Usage']}/{idx}"   (idx = original pd.iterrows index)
            original_idx = int(row["_original_idx"])
            pkl_key = f"{self.split}/{original_idx}"

            landmarks = self.landmarks_dict.get(pkl_key, None)

            if pkl_key not in self.landmarks_dict:
                # Key absent from dict entirely (shouldn't happen with a complete
                # extraction run, but guard defensively)
                logger.debug(
                    "[FER2013Dataset] PKL key '%s' not found in landmarks_dict. "
                    "Treating as failed detection (landmarks=None).",
                    pkl_key,
                )

            heatmap_P, valid_mask = self.spatial_prior_generator.generate(
                landmarks=landmarks,
                emotion_label=label,
            )

        else:
            # Baseline / inference mode — return silent zero-filled dummies.
            # Shape matches SpatialPriorGenerator default output exactly.
            heatmap_size = (
                self.spatial_prior_generator.target_size
                if self.spatial_prior_generator is not None
                else _DEFAULT_HEATMAP_SIZE
            )
            heatmap_P = torch.zeros(1, heatmap_size, heatmap_size,
                                    dtype=torch.float32)
            valid_mask = torch.zeros(1, dtype=torch.float32)

        return image, label, heatmap_P, valid_mask