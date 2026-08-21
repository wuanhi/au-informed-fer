# =============================================================================
# fer-stage1/scripts/extract_landmarks.py
#
# Stage 1 – Ablation A2: Offline 68-Point Facial Landmark Extraction
# Supports two modes via argparse:
#   --visualize : Sanity check — plots 20 random samples with annotated landmarks
#   --extract   : Full extraction — saves landmarks for the entire FER2013 dataset
# =============================================================================

import argparse
import os
import pickle
import random
import sys
from pathlib import Path

import face_alignment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Landmark indices to highlight in RED for anatomical verification:
#   21, 22 → Left & Right eyebrow inner corners  (AU4 / brow region)
#   48, 54 → Left & Right mouth corners          (AU12/AU15 / lip region)
HIGHLIGHT_POINTS = [21, 22, 48, 54, 31, 35, 57, 58, 59]
EMOTION_MAP = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(csv_path: str) -> pd.DataFrame:
    """Load the FER2013 CSV and return the full DataFrame."""
    print(f"[INFO] Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Total rows: {len(df):,}  |  Splits: {df['Usage'].value_counts().to_dict()}")
    return df


def decode_image(pixels_str: str, target_size: int = 224) -> np.ndarray:
    """
    Decode a FER2013 pixel string into a (target_size, target_size, 3) uint8 RGB array.

    Steps:
        1. Parse space-separated integers → (48, 48) grayscale array
        2. Wrap in PIL Image (mode='L')
        3. Resize to target_size × target_size with BILINEAR interpolation
        4. Convert to RGB  →  replicate single channel across R, G, B
        5. Return as numpy array (H, W, 3)
    """
    pixels = np.fromstring(pixels_str, dtype=np.uint8, sep=" ")
    img_gray = pixels.reshape(48, 48)
    pil_img = Image.fromarray(img_gray, mode="L")
    pil_img = pil_img.resize((target_size, target_size), Image.BILINEAR)
    pil_img_rgb = pil_img.convert("RGB")
    return np.array(pil_img_rgb, dtype=np.uint8)


def build_fa_detector(device: str = "cuda") -> face_alignment.FaceAlignment:
    """
    Instantiate the face_alignment detector.

    Uses SFD (S3FD) face detector — more robust than dlib on small/noisy faces.
    Falls back to CPU if CUDA is unavailable.
    """
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available — falling back to CPU. Extraction will be slow.")
        device = "cpu"
    print(f"[INFO] Initializing face_alignment on device: {device}")
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device=device,
    )
    return fa


def detect_landmarks(fa: face_alignment.FaceAlignment, img_rgb: np.ndarray) -> np.ndarray | None:
    """
    Detect 68 landmarks for a single image.

    Args:
        fa:      Initialized FaceAlignment object.
        img_rgb: (H, W, 3) uint8 RGB numpy array.

    Returns:
        (68, 2) float32 array of (x, y) landmark coordinates,
        or None if no face is detected.
    """
    preds = fa.get_landmarks(img_rgb)
    if preds is None or len(preds) == 0:
        return None
    # If multiple faces, take the first (largest) detection
    landmarks = preds[0]  # shape: (68, 2)
    return landmarks.astype(np.float32)


def make_output_dir(output_dir: str) -> Path:
    """Ensure the output directory exists and return its Path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# MODE 1: Visualize (Sanity Check)
# ---------------------------------------------------------------------------

def run_visualize(args: argparse.Namespace) -> None:
    """
    Randomly sample 20 images, detect landmarks, and plot a 4×5 annotated grid.

    Landmark rendering rules:
      - All 68 landmarks: green dots  (size = 12)
      - HIGHLIGHT_POINTS [21, 22, 48, 54]: red dots (size = 40) drawn on top
    """
    df = load_csv(args.csv_path)
    fa = build_fa_detector(args.device)
    out_dir = make_output_dir(args.output_dir)

    # --- Sample 20 random rows (across all splits) ---
    sample_df = df.sample(n=min(40, len(df)), random_state=args.seed).reset_index(drop=True)

    n_cols, n_rows = 8, 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, 18))
    fig.suptitle(
        "FER2013 Landmark Sanity Check\n"
        "🟢 All landmarks  🔴 Points [21, 22] (brow inner) & [48, 54] (mouth corners)",
        fontsize=16,
        fontweight="bold",
    )
    axes_flat = axes.flatten()

    detected_count = 0
    failed_count = 0

    for i, (_, row) in enumerate(sample_df.iterrows()):
        ax = axes_flat[i]
        img_rgb = decode_image(row["pixels"], target_size=224)
        emotion_label = EMOTION_MAP.get(int(row["emotion"]), "unknown")
        split_label = row["Usage"]

        landmarks = detect_landmarks(fa, img_rgb)

        ax.imshow(img_rgb)
        ax.axis("off")

        if landmarks is not None:
            detected_count += 1
            # Plot all 68 landmarks in green
            ax.scatter(
                landmarks[:, 0],
                landmarks[:, 1],
                s=12,
                c="lime",
                zorder=2,
                linewidths=0,
            )
            # Overlay highlighted anatomical control points in red (larger)
            highlight = landmarks[HIGHLIGHT_POINTS]
            ax.scatter(
                highlight[:, 0],
                highlight[:, 1],
                s=60,
                c="red",
                zorder=3,
                linewidths=0.5,
                edgecolors="white",
            )
            status_text = f"{emotion_label} | {split_label}"
            ax.set_title(status_text, fontsize=8, color="black", pad=2)
        else:
            failed_count += 1
            ax.set_title(
                f"{emotion_label} | {split_label}\n⚠️ No face detected",
                fontsize=8,
                color="red",
                pad=2,
            )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = out_dir / "landmark_sanity_check.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[RESULT] Visualize complete.")
    print(f"         Detected : {detected_count}/20")
    print(f"         Failed   : {failed_count}/20")
    print(f"         Saved to : {out_path.resolve()}")


# ---------------------------------------------------------------------------
# MODE 2: Extract (Full Dataset)
# ---------------------------------------------------------------------------

def run_extract(args: argparse.Namespace) -> None:
    """
    Iterate through the entire FER2013 CSV and extract 68 landmarks per image.

    Output format (pkl):
        dict[str, np.ndarray | None]
        Key   → "{Usage}/{row_index}"   e.g. "Training/0", "PublicTest/1234"
        Value → (68, 2) float32 numpy array, or None if detection failed
    """
    df = load_csv(args.csv_path)
    fa = build_fa_detector(args.device)
    out_dir = make_output_dir(args.output_dir)

    results: dict[str, np.ndarray | None] = {}
    total = len(df)
    failed = 0
    detected = 0

    print(f"\n[INFO] Starting extraction for {total:,} images...")

    for idx, row in tqdm(df.iterrows(), total=total, desc="Extracting landmarks", unit="img"):
        key = f"{row['Usage']}/{idx}"
        img_rgb = decode_image(row["pixels"], target_size=224)
        landmarks = detect_landmarks(fa, img_rgb)

        if landmarks is None:
            failed += 1
            results[key] = None
        else:
            detected += 1
            results[key] = landmarks  # (68, 2) float32

    # --- Save to pickle ---
    out_path = out_dir / "fer2013_landmarks.pkl"
    with open(str(out_path), "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    # --- Summary ---
    detection_rate = (detected / total) * 100
    print(f"\n[RESULT] Extraction complete.")
    print(f"         Total    : {total:,}")
    print(f"         Detected : {detected:,}  ({detection_rate:.2f}%)")
    print(f"         Failed   : {failed:,}  ({100 - detection_rate:.2f}%)")
    print(f"         Saved to : {out_path.resolve()}")
    print(f"\n[PKL structure] dict['{{}}/idx'] → np.ndarray(68, 2) | None")
    print(f"[PKL size]      {out_path.stat().st_size / 1024 / 1024:.2f} MB")


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FER2013 Stage 1-A2: Offline 68-point landmark extraction.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # --- Mode (mutually exclusive) ---
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--visualize",
        action="store_true",
        help="Sanity check: plot 20 random samples with annotated landmarks.",
    )
    mode_group.add_argument(
        "--extract",
        action="store_true",
        help="Full extraction: process entire FER2013 dataset and save to .pkl.",
    )

    # --- Paths ---
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/fer2013.csv",
        help="Path to the FER2013 CSV file (default: data/fer2013.csv).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save outputs (default: outputs/).",
    )

    # --- Hardware ---
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for face_alignment inference (default: cuda).",
    )

    # --- Misc ---
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sample selection in --visualize (default: 42).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    if args.visualize:
        run_visualize(args)
    elif args.extract:
        run_extract(args)