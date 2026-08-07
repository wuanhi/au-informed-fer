import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.utils.config import load_config
from src.utils.seed import set_seed

from src.data.dataloader import build_dataloaders

from src.models.convnext_backbone import build_convnext_tiny

from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.training.trainer import train_one_epoch, evaluate_one_epoch
from src.training.checkpoint import save_checkpoint

from src.losses.classification import build_classification_loss
from src.utils.logger import append_epoch_log

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True
    )

    parser.add_argument(
        "--data-config",
        type=str,
        required=True
    )

    args = parser.parse_args()

    # =========================
    # Load config
    # =========================
    cfg = load_config(args.config)
    data_cfg = load_config(args.data_config)

    seed = cfg["training"]["seed"]
    epochs = cfg["training"]["epochs"]
    use_amp = cfg["training"].get("amp", True)

    set_seed(seed)

    # =========================
    # Device
    # =========================
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # =========================
    # Data
    # =========================
    train_loader, val_loader, test_loader = build_dataloaders(
        data_cfg,
        cfg
    )

    print(f"Train: {len(train_loader.dataset)}")
    print(f"Val:   {len(val_loader.dataset)}")
    print(f"Test:  {len(test_loader.dataset)}")

    # =========================
    # Model
    # =========================
    model_cfg = cfg["model"]

    if model_cfg["name"] == "convnext_tiny":
        model = build_convnext_tiny(
            num_classes=model_cfg["num_classes"],
            pretrained=model_cfg["pretrained"]
        )
    else:
        raise ValueError(
            f"Unsupported model: {model_cfg['name']}"
        )

    model = model.to(device)

    # =========================
    # Training components
    # =========================
    loss_fn = build_classification_loss(cfg)

    optimizer = build_optimizer(
        model,
        cfg
    )

    scheduler = build_scheduler(
        optimizer,
        cfg
    )

    # =========================
    # Output
    # =========================
    run_name = Path(args.config).stem

    output_dir = Path("outputs") / run_name
    checkpoint_dir = Path("checkpoints") / run_name

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    best_macro_f1 = -1.0

    amp_enabled = use_amp and device.type == "cuda"

    scaler = torch.amp.GradScaler(
    "cuda",
    enabled=amp_enabled
    )

    # =========================
    # Training loop
    # =========================
    for epoch in range(1, epochs + 1):

        print()
        print("=" * 60)
        print(f"Epoch {epoch}/{epochs}")
        print("=" * 60)

        train_metrics = train_one_epoch(
        model=model,
        dataloader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        scaler=scaler,
        use_amp=use_amp
        )

        val_metrics = evaluate_one_epoch(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
            use_amp=use_amp
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        append_epoch_log(
        path=output_dir / "history.csv",
        epoch=epoch,
        lr=current_lr,
        train_metrics=train_metrics,
        val_metrics=val_metrics
        )
        print(
            f"Train | "
            f"Loss: {train_metrics['loss']:.4f} | "
            f"Acc: {train_metrics['accuracy']:.4f} | "
            f"Macro-F1: {train_metrics['macro_f1']:.4f}"
        )

        print(
            f"Val   | "
            f"Loss: {val_metrics['loss']:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f} | "
            f"Macro-F1: {val_metrics['macro_f1']:.4f}"
        )

        print(
            f"LR: {current_lr:.8f}"
        )

        # =========================
        # Best checkpoint
        # =========================
        if val_metrics["macro_f1"] > best_macro_f1:

            best_macro_f1 = val_metrics["macro_f1"]

            save_checkpoint(
                path=checkpoint_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_macro_f1,
                config=cfg
            )

            print(
                f"Saved BEST checkpoint "
                f"(Macro-F1={best_macro_f1:.4f})"
            )

        # =========================
        # Last checkpoint
        # =========================
        save_checkpoint(
            path=checkpoint_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=best_macro_f1,
            config=cfg
        )

    print()
    print("=" * 60)
    print("Training completed")
    print(f"Best validation Macro-F1: {best_macro_f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()