# =============================================================================
# fer-stage1/src/data/spatial_prior_generator.py
#
# Stage 1 – Ablation A2: Semantic Prior Heatmap Generator
#
# Converts offline-extracted 68-point facial landmarks into a (1, 14, 14)
# Semantic Prior Heatmap P, conditioned on the emotion label via AU activity.
#
# Design constraints:
#   - Gaussian accumulation uses torch.max (never addition) to prevent
#     intensity explosion when landmarks or AUs spatially overlap.
#   - AU activity gate: presence_rate >= 0.15 (configurable via threshold).
#   - Neutral (label=6) and failed detections (landmarks=None) yield a
#     zero heatmap with valid_mask=0, signalling the loss to ignore this sample.
# =============================================================================

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Emotion index → column name in the presence-rate CSV
# (matches FER2013 label order; neutral=6 is handled as a base case)
# ---------------------------------------------------------------------------
_EMOTION_IDX_TO_COL: dict[int, str] = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sadness",
    5: "surprise",
    # 6: "neutral" — excluded; yields zero heatmap
}

# Source image spatial scale assumed by the landmark extractor
_LANDMARK_SPATIAL_SCALE: int = 224


class SpatialPriorGenerator:
    """
    Converts 68-point facial landmarks (extracted offline) into a
    Semantic Prior Heatmap P of shape (1, target_size, target_size).

    Args:
        au_mapping_path  (str | Path): Path to the AU→landmark-indices JSON.
                                       Expected schema:
                                       { "AU1": {"landmarks": [21, 22], ...}, ... }
        au_presence_path (str | Path): Path to the AU×emotion presence-rate CSV.
                                       Expected columns: au, anger, disgust,
                                       fear, happy, sadness, surprise.
        target_size      (int)       : Spatial size of the output heatmap
                                       (default 14 for a 14×14 grid).
        sigma            (float)     : Standard deviation of each Gaussian dot
                                       in heatmap-pixel units (default 1.0).
        presence_threshold (float)  : Minimum presence_rate to treat an AU as
                                       active for a given emotion (default 0.15).
    """

    def __init__(
        self,
        au_mapping_path: str | Path,
        au_presence_path: str | Path,
        target_size: int = 14,
        sigma: float = 1.0,
        presence_threshold: float = 0.15,
    ) -> None:
        self.target_size = target_size
        self.sigma = sigma
        self.presence_threshold = presence_threshold

        # Scale factor: landmark coords are in 224×224 space
        self._scale: float = _LANDMARK_SPATIAL_SCALE / target_size

        # ------------------------------------------------------------------
        # Load AU → landmark-indices mapping
        # ------------------------------------------------------------------
        au_mapping_path = Path(au_mapping_path)
        if not au_mapping_path.exists():
            raise FileNotFoundError(
                f"[SpatialPriorGenerator] AU mapping JSON not found: {au_mapping_path}"
            )
        with open(au_mapping_path, "r", encoding="utf-8") as f:
            raw_mapping: dict = json.load(f)

        # Normalise to { "AU1": [21, 22], ... }
        self._au_to_landmark_indices: dict[str, list[int]] = {
            au_name: info["landmarks"]
            for au_name, info in raw_mapping.items()
        }

        # ------------------------------------------------------------------
        # Load AU × emotion presence-rate table
        # Build: { emotion_idx: { "AU1": rate, "AU4": rate, ... } }
        #        keeping only AUs present in the mapping JSON
        # ------------------------------------------------------------------
        au_presence_path = Path(au_presence_path)
        if not au_presence_path.exists():
            raise FileNotFoundError(
                f"[SpatialPriorGenerator] AU presence CSV not found: {au_presence_path}"
            )
        df = pd.read_csv(au_presence_path)
        df = df.set_index("au")

        self._active_aus: dict[int, dict[str, float]] = {}

        for emotion_idx, col_name in _EMOTION_IDX_TO_COL.items():
            if col_name not in df.columns:
                logger.warning(
                    "[SpatialPriorGenerator] Column '%s' not found in presence CSV "
                    "(emotion_idx=%d). Using empty AU set.",
                    col_name,
                    emotion_idx,
                )
                self._active_aus[emotion_idx] = {}
                continue

            active: dict[str, float] = {}
            for au_name, rate in df[col_name].items():
                # Gate 1: AU must appear in the landmark mapping JSON
                if au_name not in self._au_to_landmark_indices:
                    continue
                # Gate 2: presence rate must clear the activity threshold
                if rate >= self.presence_threshold:
                    active[au_name] = float(rate)

            self._active_aus[emotion_idx] = active
            logger.debug(
                "[SpatialPriorGenerator] emotion=%d (%s) → %d active AUs: %s",
                emotion_idx,
                col_name,
                len(active),
                list(active.keys()),
            )

        # ------------------------------------------------------------------
        # Pre-compute the coordinate meshgrid (shape: target_size × target_size)
        # for fast Gaussian evaluation — reused on every generate() call.
        # grid_y[i, j] = i,  grid_x[i, j] = j
        # ------------------------------------------------------------------
        coords = torch.arange(target_size, dtype=torch.float32)
        self._grid_y, self._grid_x = torch.meshgrid(coords, coords, indexing="ij")
        # Shape: (target_size, target_size) — kept on CPU; moved to device in generate()

        logger.info(
            "[SpatialPriorGenerator] Initialised — target_size=%d, sigma=%.2f, "
            "threshold=%.2f, scale=%.4f",
            self.target_size,
            self.sigma,
            self.presence_threshold,
            self._scale,
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _gaussian_dot(
        self,
        cx: float,
        cy: float,
        amplitude: float,
    ) -> torch.Tensor:
        """
        Evaluate a 2D isotropic Gaussian centred at (cx, cy) over the full grid.

        Formula:
            G(i, j) = amplitude * exp( -( (j - cx)^2 + (i - cy)^2 ) / (2σ²) )

        Note: cx is the column (x-axis / width),
              cy is the row    (y-axis / height).

        Returns:
            Tensor of shape (target_size, target_size), values in [0, amplitude].
        """
        two_sigma_sq = 2.0 * (self.sigma ** 2)
        dist_sq = (self._grid_x - cx) ** 2 + (self._grid_y - cy) ** 2
        return amplitude * torch.exp(-dist_sq / two_sigma_sq)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def generate(
        self,
        landmarks: Optional[np.ndarray],
        emotion_label: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate the Semantic Prior Heatmap for one sample.

        Args:
            landmarks     : (68, 2) float32 numpy array of (x, y) landmark
                            coordinates in 224×224 pixel space, or None if
                            face detection failed during offline extraction.
            emotion_label : Integer emotion class index in [0, 6] following
                            the FER2013 convention:
                              0=Angry, 1=Disgust, 2=Fear, 3=Happy,
                              4=Sad,   5=Surprise, 6=Neutral.

        Returns:
            heatmap_P  (torch.Tensor): Shape (1, target_size, target_size).
                                       All values in [0, 1].
            valid_mask (torch.Tensor): Shape (1,).
                                       1 → valid heatmap (use in loss).
                                       0 → invalid/neutral (skip in loss).
        """
        # ------------------------------------------------------------------
        # Base case: missing landmarks OR neutral emotion → zero heatmap
        # ------------------------------------------------------------------
        if landmarks is None or emotion_label == 6:
            heatmap_P = torch.zeros(1, self.target_size, self.target_size,
                                    dtype=torch.float32)
            valid_mask = torch.zeros(1, dtype=torch.float32)
            return heatmap_P, valid_mask

        valid_mask = torch.ones(1, dtype=torch.float32)

        # ------------------------------------------------------------------
        # Retrieve active AUs for this emotion
        # ------------------------------------------------------------------
        active_aus: dict[str, float] = self._active_aus.get(emotion_label, {})

        if not active_aus:
            # No AUs cleared the threshold for this emotion — return zero map
            logger.debug(
                "[SpatialPriorGenerator] emotion_label=%d has no active AUs; "
                "returning zero heatmap with valid_mask=1.",
                emotion_label,
            )
            heatmap_P = torch.zeros(1, self.target_size, self.target_size,
                                    dtype=torch.float32)
            return heatmap_P, valid_mask

        # ------------------------------------------------------------------
        # Scale landmarks from 224×224 down to target_size × target_size
        # landmarks[:, 0] = x (column), landmarks[:, 1] = y (row)
        # ------------------------------------------------------------------
        scaled = landmarks / self._scale  # (68, 2) float32 — still numpy

        # ------------------------------------------------------------------
        # Accumulate Gaussians using torch.max (never addition)
        # ------------------------------------------------------------------
        # Running accumulator — starts at zero, shape (target_size, target_size)
        heatmap = torch.zeros(self.target_size, self.target_size,
                              dtype=torch.float32)

        for au_name, presence_rate in active_aus.items():
            landmark_indices: list[int] = self._au_to_landmark_indices[au_name]

            for lm_idx in landmark_indices:
                # Boundary guard: skip invalid indices (malformed landmarks)
                if lm_idx < 0 or lm_idx >= len(scaled):
                    logger.warning(
                        "[SpatialPriorGenerator] AU %s references landmark index %d "
                        "which is out of bounds for a (68, 2) array. Skipping.",
                        au_name,
                        lm_idx,
                    )
                    continue

                cx = float(scaled[lm_idx, 0])  # column (x)
                cy = float(scaled[lm_idx, 1])  # row    (y)

                # Skip landmarks that fall outside the heatmap canvas
                if not (0.0 <= cx < self.target_size and 0.0 <= cy < self.target_size):
                    logger.debug(
                        "[SpatialPriorGenerator] Landmark %d (%.2f, %.2f) is outside "
                        "the %dx%d heatmap after scaling. Skipping.",
                        lm_idx, cx, cy, self.target_size, self.target_size,
                    )
                    continue

                # Gaussian dot weighted by this AU's presence rate
                gauss = self._gaussian_dot(cx, cy, amplitude=presence_rate)

                # ✅ CRITICAL: use torch.max — never addition
                # Prevents intensity explosion when landmarks or AUs overlap
                heatmap = torch.max(heatmap, gauss)

        # Add channel dimension → (1, target_size, target_size)
        heatmap_P = heatmap.unsqueeze(0)

        return heatmap_P, valid_mask