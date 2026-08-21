# =============================================================================
# fer-stage1/scripts/train_a2.py
#
# Stage 1 – Ablation A2: Training Script
#
# Key differences from scripts/train.py (A0/A1 baseline):
#
#   1. Loads offline fer2013_landmarks.pkl and initialises SpatialPriorGenerator.
#   2. Passes both into build_dataloaders → FER2013Dataset yields 4-tuples:
#        (images, labels, heatmaps_P, valid_masks)
#   3. Model: ConvNeXt_A2 (pure ConvNeXt-Tiny + Attention Branch, NO STN).
#      Forward returns (logits, A) where A is (B, 1, 14, 14).
#   4. Combined loss per batch:
#        loss_cls  = CrossEntropy(logits, labels)          [label-smoothed]
#        loss_sp   = masked MSE(A, heatmap_P, valid_mask)  [spatial alignment]
#        loss_total = loss_cls + lambda_sp * loss_sp
#   5. Self-contained A2 epoch loops (_train_one_epoch_a2, etc.) to avoid
#      breaking the shared trainer.py which unpacks (inputs, labels) only.
#   6. history_a2.csv logs loss_cls, loss_sp, and loss_total per epoch.
#
# Usage:
#   python scripts/train_a2.py \
#       --config        configs/A2/convnext_tiny_a2.yaml \
#       --data-config   configs/data/fer2013.yaml \
#       --landmarks-path outputs/fer2013_landmarks.pkl \
#       --au-mapping    data/CK+/au_landmark_mapping.json \
#       --au-presence   datasets/AU_Emotion_Matrix_PresenceRate.csv \
#       --lambda-sp     0.1
# =============================================================================

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from ema_pytorch import EMA
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project root on sys.path (mirrors train.py)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from src.data.dataloader import build_dataloaders
from src.data.spatial_prior_generator import SpatialPriorGenerator
from src.evaluation.metrics import compute_metrics
from src.losses.classification import build_classification_loss
from src.models.convnext_a2 import build_convnext_a2
from src.training.checkpoint import save_checkpoint
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.utils.config import load_config
from src.utils.seed import set_seed


# ===========================================================================
# A2-specific epoch functions
# (Cannot reuse trainer.py — it unpacks (inputs, labels) only)
# ===========================================================================

def _compute_spatial_loss(
    A: torch.Tensor,
    heatmaps_P: torch.Tensor,
    valid_masks: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Masked MSE between predicted attention map A and prior heatmap P.

    Steps:
        1. Compute element-wise MSE (no reduction):
               mse  shape: (B, 1, H, W)
        2. Average over the spatial dimensions H and W:
               mse_per_sample  shape: (B, 1)
        3. Squeeze to (B,) and multiply by valid_mask (B,):
               masked_mse  shape: (B,)
           Samples where valid_mask=0 (neutral / no landmarks) contribute 0.
        4. Sum and divide by the count of valid samples (+ eps).
           Using sum-then-divide rather than plain .mean() ensures that a
           batch full of neutral/failed samples returns 0 instead of NaN.

    Args:
        A           (B, 1, H, W): predicted attention map in [0, 1].
        heatmaps_P  (B, 1, H, W): prior heatmap from SpatialPriorGenerator.
        valid_masks (B, 1)       : 1.0 for valid samples, 0.0 otherwise.
        eps         float        : small constant to avoid division by zero.

    Returns:
        loss_sp (scalar Tensor)
    """
    # (B, 1, H, W)
    mse = F.mse_loss(A, heatmaps_P, reduction="none")

    # Average over H and W → (B, 1)
    mse_per_sample = mse.mean(dim=[-2, -1])

    # Flatten valid_masks from (B, 1) → (B,) to match mse_per_sample squeezed
    mask = valid_masks.squeeze(1)           # (B,)
    mse_squeezed = mse_per_sample.squeeze(1)  # (B,)

    # Zero out invalid samples, then average over valid ones
    masked_mse = mse_squeezed * mask        # (B,)
    n_valid = mask.sum() + eps
    loss_sp = masked_mse.sum() / n_valid

    return loss_sp


def _train_one_epoch_a2(
    model: torch.nn.Module,
    dataloader,
    optimizer,
    scheduler,
    loss_fn,
    device: torch.device,
    scaler,
    ema,
    use_amp: bool,
    gradient_clip: float,
    gradient_accumulation_steps: int,
    lambda_sp: float,
) -> dict:
    """
    One training epoch for ConvNeXt_A2.

    Batch unpacking: (images, labels, heatmaps_P, valid_masks)
    Model forward  : (logits, A)
    Loss           : loss_cls + lambda_sp * loss_sp

    Returns a metrics dict identical to trainer.py's train_one_epoch output,
    extended with 'loss_cls', 'loss_sp', and 'loss_total' keys.
    """
    model.train()

    amp_enabled = use_amp and device.type == "cuda"

    batch_losses_total = []
    batch_losses_cls   = []
    batch_losses_sp    = []
    batch_accuracies   = []
    all_targets        = []
    all_predictions    = []

    pbar = tqdm(dataloader, desc="Training")

    for batch_idx, (
        images,
        labels,
        heatmaps_P,
        valid_masks,
    ) in enumerate(pbar):

        images     = images.to(device)
        labels     = labels.to(device)
        heatmaps_P = heatmaps_P.to(device)
        valid_masks = valid_masks.to(device)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            logits, A = model(images)

            # --- Classification loss (label-smoothed CrossEntropy) ---
            loss_cls = loss_fn(logits, labels)

            # --- Spatial alignment loss (masked MSE) ---
            loss_sp = _compute_spatial_loss(A, heatmaps_P, valid_masks)

            # --- Combined loss ---
            loss = loss_cls + lambda_sp * loss_sp

        predictions = torch.argmax(logits, dim=1)

        scaler.scale(loss).backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Unscale before clipping so max_norm is in real-gradient space
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                gradient_clip,
            )

            scaler.step(optimizer)
            optimizer.zero_grad(set_to_none=True)
            scaler.update()

            ema.update()

            # Scheduler is stepped per optimizer update (matches baseline)
            scheduler.step()

        batch_accuracy = (
            (predictions == labels).sum().item() / labels.size(0)
        )

        batch_losses_total.append(loss.item())
        batch_losses_cls.append(loss_cls.item())
        batch_losses_sp.append(loss_sp.item())
        batch_accuracies.append(batch_accuracy)

        all_targets.extend(labels.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

        pbar.set_postfix(
            {
                "loss": f"{np.mean(batch_losses_total):.4f}",
                "cls":  f"{np.mean(batch_losses_cls):.4f}",
                "sp":   f"{np.mean(batch_losses_sp):.4f}",
                "acc":  f"{np.mean(batch_accuracies) * 100:.1f}%",
            }
        )

    metrics = compute_metrics(all_targets, all_predictions)

    # EmoNeXt source averages batch losses and accuracies
    metrics["loss"]       = float(np.mean(batch_losses_total))
    metrics["loss_cls"]   = float(np.mean(batch_losses_cls))
    metrics["loss_sp"]    = float(np.mean(batch_losses_sp))
    metrics["loss_total"] = metrics["loss"]
    metrics["accuracy"]   = float(np.mean(batch_accuracies))

    return metrics


@torch.no_grad()
def _evaluate_one_epoch_a2(
    model: torch.nn.Module,
    dataloader,
    loss_fn,
    device: torch.device,
    use_amp: bool,
    lambda_sp: float,
) -> dict:
    """
    One validation epoch for ConvNeXt_A2.

    Returns the same extended metrics dict as _train_one_epoch_a2.
    """
    model.eval()

    amp_enabled = use_amp and device.type == "cuda"

    batch_losses_total = []
    batch_losses_cls   = []
    batch_losses_sp    = []
    all_targets        = []
    all_predictions    = []

    pbar = tqdm(dataloader, desc="Validation")

    for (
        images,
        labels,
        heatmaps_P,
        valid_masks,
    ) in pbar:
        images      = images.to(device)
        labels      = labels.to(device)
        heatmaps_P  = heatmaps_P.to(device)
        valid_masks = valid_masks.to(device)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            logits, A = model(images)

            loss_cls = loss_fn(logits, labels)
            loss_sp  = _compute_spatial_loss(A, heatmaps_P, valid_masks)
            loss     = loss_cls + lambda_sp * loss_sp

        predictions = torch.argmax(logits, dim=1)

        batch_losses_total.append(loss.item())
        batch_losses_cls.append(loss_cls.item())
        batch_losses_sp.append(loss_sp.item())

        all_targets.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    metrics = compute_metrics(all_targets, all_predictions)

    metrics["loss"]       = float(np.mean(batch_losses_total))
    metrics["loss_cls"]   = float(np.mean(batch_losses_cls))
    metrics["loss_sp"]    = float(np.mean(batch_losses_sp))
    metrics["loss_total"] = metrics["loss"]

    return metrics


@torch.no_grad()
def _evaluate_test_ema_a2(
    ema,
    dataloader,
    device: torch.device,
    use_amp: bool,
) -> dict:
    """
    Final test evaluation using the EMA model.

    Handles TenCrop batches (B, 10, C, H, W) exactly as trainer.py does.
    ConvNeXt_A2 returns (logits, A); we discard A during inference.
    """
    ema.eval()

    amp_enabled = use_amp and device.type == "cuda"

    all_targets     = []
    all_predictions = []

    pbar = tqdm(dataloader, desc="Testing")

    for (
        images,
        labels,
        _heatmaps_P,    # not used during test evaluation
        _valid_masks,
    ) in pbar:
        # TenCrop: images is (B, 10, C, H, W)
        bs, ncrops, c, h, w = images.shape
        images = images.view(-1, c, h, w)  # (B*10, C, H, W)

        images = images.to(device)
        labels = labels.to(device)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            # EMA wraps ConvNeXt_A2; forward returns (logits, A)
            logits, _A = ema(images)

        # Reshape and average over crops: (B*10, num_classes) → (B, num_classes)
        logits = logits.view(bs, ncrops, -1)
        outputs_avg = logits.mean(dim=1)   # (B, num_classes)

        predictions = torch.argmax(outputs_avg, dim=1)

        all_targets.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    return compute_metrics(all_targets, all_predictions)


# ===========================================================================
# A2-specific CSV logger
# (Extends logger.py's schema with loss_cls, loss_sp, loss_total columns)
# ===========================================================================

def _append_epoch_log_a2(
    path: Path,
    epoch: int,
    lr: float,
    train_metrics: dict,
    val_metrics: dict,
) -> None:
    """
    Append one epoch row to history_a2.csv.

    Columns (superset of A0/A1 history.csv for easy ablation comparison):
        epoch, lr,
        train_loss_total, train_loss_cls, train_loss_sp,
        train_accuracy, train_macro_f1,
        val_loss_total, val_loss_cls, val_loss_sp,
        val_accuracy, val_macro_f1, val_balanced_accuracy
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists()

    row = {
        "epoch":                 epoch,
        "lr":                    lr,
        # --- Train ---
        "train_loss_total":      train_metrics["loss_total"],
        "train_loss_cls":        train_metrics["loss_cls"],
        "train_loss_sp":         train_metrics["loss_sp"],
        "train_accuracy":        train_metrics["accuracy"],
        "train_macro_f1":        train_metrics["macro_f1"],
        # --- Validation ---
        "val_loss_total":        val_metrics["loss_total"],
        "val_loss_cls":          val_metrics["loss_cls"],
        "val_loss_sp":           val_metrics["loss_sp"],
        "val_accuracy":          val_metrics["accuracy"],
        "val_macro_f1":          val_metrics["macro_f1"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
    }

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FER2013 Stage 1 – Ablation A2 Training",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # --- Config paths ---
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to the A2 experiment YAML (e.g. configs/A2/convnext_tiny_a2.yaml).",
    )
    parser.add_argument(
        "--data-config",
        required=True,
        type=str,
        help="Path to the data YAML (e.g. configs/data/fer2013.yaml).",
    )

    # --- A2-specific paths ---
    parser.add_argument(
        "--landmarks-path",
        type=str,
        default="outputs/fer2013_landmarks.pkl",
        help="Path to the offline-extracted fer2013_landmarks.pkl file.\n"
             "Default: outputs/fer2013_landmarks.pkl",
    )
    parser.add_argument(
        "--au-mapping",
        type=str,
        default="data/CK+/au_landmark_mapping.json",
        help="Path to the AU→landmark-indices JSON.\n"
             "Default: data/CK+/au_landmark_mapping.json",
    )
    parser.add_argument(
        "--au-presence",
        type=str,
        default="datasets/AU_Emotion_Matrix_PresenceRate.csv",
        help="Path to the AU×emotion presence-rate CSV.\n"
             "Default: datasets/AU_Emotion_Matrix_PresenceRate.csv",
    )

    # --- A2 loss hyperparameter ---
    parser.add_argument(
        "--lambda-sp",
        type=float,
        default=0.1,
        help="Weight for the spatial alignment loss term.\n"
             "loss_total = loss_cls + lambda_sp * loss_sp\n"
             "Default: 0.1",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load configs
    # ------------------------------------------------------------------
    cfg      = load_config(args.config)
    data_cfg = load_config(args.data_config)

    lambda_sp: float = args.lambda_sp

    set_seed(cfg["training"]["seed"])

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # ------------------------------------------------------------------
    # A2 Step 1: Load offline landmarks dict
    # ------------------------------------------------------------------
    landmarks_path = Path(args.landmarks_path)
    if not landmarks_path.exists():
        raise FileNotFoundError(
            f"[train_a2] Landmarks PKL not found: {landmarks_path}\n"
            "Run:  python scripts/extract_landmarks.py --extract"
        )
    print(f"[A2] Loading landmarks from: {landmarks_path}")
    with open(landmarks_path, "rb") as f:
        landmarks_dict = pickle.load(f)
    print(f"[A2] Landmarks loaded — {len(landmarks_dict):,} entries.")

    # ------------------------------------------------------------------
    # A2 Step 2: Initialise SpatialPriorGenerator
    # ------------------------------------------------------------------
    print(f"[A2] Initialising SpatialPriorGenerator ...")
    prior_gen = SpatialPriorGenerator(
        au_mapping_path=args.au_mapping,
        au_presence_path=args.au_presence,
        target_size=14,
        sigma=1.0,
    )

    # ------------------------------------------------------------------
    # A2 Step 3: Build DataLoaders (thread prior args)
    # ------------------------------------------------------------------
    (
        train_loader,
        val_loader,
        test_loader,
    ) = build_dataloaders(
        data_cfg,
        cfg,
        spatial_prior_generator=prior_gen,
        landmarks_dict=landmarks_dict,
    )

    print(f"Train: {len(train_loader.dataset):,}")
    print(f"Val:   {len(val_loader.dataset):,}")
    print(f"Test:  {len(test_loader.dataset):,}")

    # ------------------------------------------------------------------
    # A2 Step 4: Model — pure ConvNeXt-Tiny + Attention Branch (no STN)
    # ------------------------------------------------------------------
    model = build_convnext_a2(
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
        drop_path_rate=cfg["model"]["drop_path_rate"],
    ).to(device)

    # ------------------------------------------------------------------
    # Loss, Optimiser, Scheduler, AMP, EMA — identical to baseline
    # ------------------------------------------------------------------
    loss_fn = build_classification_loss(cfg)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    use_amp = cfg["training"]["amp"]
    amp_enabled = use_amp and device.type == "cuda"

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    ema = EMA(
        model,
        beta=cfg["ema"]["beta"],
        update_every=cfg["ema"]["update_every"],
    ).to(device)

    # ------------------------------------------------------------------
    # Output / checkpoint directories (derived from config stem)
    # ------------------------------------------------------------------
    run_name       = Path(args.config).stem
    output_dir     = Path("outputs")     / run_name
    checkpoint_dir = Path("checkpoints") / run_name

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[A2] lambda_sp = {lambda_sp}")
    print(f"[A2] Run name  : {run_name}")
    print(f"[A2] Outputs   : {output_dir}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_accuracy  = 0.0
    early_stop_counter = 0
    patience  = cfg["training"]["early_stopping_patience"]
    max_epochs = cfg["training"]["epochs"]

    for epoch in range(1, max_epochs + 1):
        print(f"\n# Epoch {epoch}/{max_epochs}")

        train_metrics = _train_one_epoch_a2(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
            scaler=scaler,
            ema=ema,
            use_amp=use_amp,
            gradient_clip=cfg["training"]["gradient_clip"],
            gradient_accumulation_steps=(
                cfg["training"]["gradient_accumulation_steps"]
            ),
            lambda_sp=lambda_sp,
        )

        val_metrics = _evaluate_one_epoch_a2(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
            use_amp=use_amp,
            lambda_sp=lambda_sp,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        # --- Console output ---
        print(
            f"Train  | loss={train_metrics['loss_total']:.4f}  "
            f"cls={train_metrics['loss_cls']:.4f}  "
            f"sp={train_metrics['loss_sp']:.4f}  "
            f"acc={train_metrics['accuracy']:.4f}  "
            f"f1={train_metrics['macro_f1']:.4f}"
        )
        print(
            f"Val    | loss={val_metrics['loss_total']:.4f}  "
            f"cls={val_metrics['loss_cls']:.4f}  "
            f"sp={val_metrics['loss_sp']:.4f}  "
            f"acc={val_metrics['accuracy']:.4f}  "
            f"f1={val_metrics['macro_f1']:.4f}"
        )
        print(f"LR: {current_lr:.8f}")

        # --- CSV logging ---
        _append_epoch_log_a2(
            path=output_dir / "history_a2.csv",
            epoch=epoch,
            lr=current_lr,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )

        # --- Checkpoint (select by val accuracy — matches EmoNeXt) ---
        val_accuracy = val_metrics["accuracy"]

        if val_accuracy > best_val_accuracy:
            best_val_accuracy  = val_accuracy
            early_stop_counter = 0

            save_checkpoint(
                path=checkpoint_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                ema=ema,
                scaler=scaler,
                scheduler=scheduler,
                best_metric=best_val_accuracy,
                epoch=epoch,
                config=cfg,
            )
            print(
                f"Saved new best checkpoint "
                f"(Val Acc={best_val_accuracy:.4f})"
            )

        else:
            early_stop_counter += 1

            if early_stop_counter >= patience:
                print(
                    f"Validation accuracy did not improve for "
                    f"{patience} epochs. Early stopping."
                )
                break

    # ------------------------------------------------------------------
    # Final test evaluation
    # IMPORTANT: Match EmoNeXt source — do NOT reload best checkpoint.
    # Evaluate the current EMA model directly.
    # ------------------------------------------------------------------
    test_metrics = _evaluate_test_ema_a2(
        ema=ema,
        dataloader=test_loader,
        device=device,
        use_amp=use_amp,
    )

    print("\n# Test")
    print(f"Test Accuracy:          {test_metrics['accuracy']:.4f}")
    print(f"Test Macro-F1:          {test_metrics['macro_f1']:.4f}")
    print(f"Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")


if __name__ == "__main__":
    main()