import sys
import argparse

from pathlib import Path

import torch

from ema_pytorch import EMA


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logger import append_epoch_log
from src.models.convnext_stn import build_stn_convnext_tiny
from src.data.dataloader import (
    build_dataloaders,
)

from src.models.convnext_backbone import (
    build_convnext_tiny,
)

from src.losses.classification import (
    build_classification_loss,
)

from src.training.optimizer import (
    build_optimizer,
)

from src.training.scheduler import (
    build_scheduler,
)

from src.training.trainer import (
    train_one_epoch,
    evaluate_one_epoch,
    evaluate_test_ema,
)

from src.training.checkpoint import (
    save_checkpoint,
)

def build_model(model_cfg):
    name = model_cfg["name"]

    if name == "convnext_tiny":
        return build_convnext_tiny(
            num_classes=model_cfg["num_classes"],
            pretrained=model_cfg["pretrained"],
            drop_path_rate=model_cfg[
                "drop_path_rate"
            ],
        )

    if name == "stn_convnext_tiny":
        return build_stn_convnext_tiny(
            num_classes=model_cfg["num_classes"],
            pretrained=model_cfg["pretrained"],
            drop_path_rate=model_cfg[
                "drop_path_rate"
            ],
        )

    raise ValueError(
        f"Unsupported model: {name}"
    )

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--data-config",
        required=True,
        type=str,
    )

    args = parser.parse_args()

    cfg = load_config(
        args.config
    )

    data_cfg = load_config(
        args.data_config
    )

    set_seed(
        cfg["training"]["seed"]
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    (
        train_loader,
        val_loader,
        test_loader,
    ) = build_dataloaders(
        data_cfg,
        cfg,
    )

    print(
        f"Train: {len(train_loader.dataset)}"
    )

    print(
        f"Val:   {len(val_loader.dataset)}"
    )

    print(
        f"Test:  {len(test_loader.dataset)}"
    )

    model = build_model(
        cfg["model"]
    ).to(device)


    loss_fn = (
        build_classification_loss(
            cfg
        )
    )

    optimizer = build_optimizer(
        model,
        cfg,
    )

    scheduler = build_scheduler(
        optimizer,
        cfg,
    )

    use_amp = cfg["training"]["amp"]

    amp_enabled = (
        use_amp
        and device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    ema = EMA(
        model,
        beta=cfg["ema"]["beta"],
        update_every=cfg["ema"][
            "update_every"
        ],
    ).to(device)

    run_name = Path(
        args.config
    ).stem

    output_dir = (
        Path("outputs")
        / run_name
    )

    checkpoint_dir = (
        Path("checkpoints")
        / run_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_accuracy = 0.0

    early_stop_counter = 0

    patience = cfg["training"][
        "early_stopping_patience"
    ]

    max_epochs = cfg["training"][
        "epochs"
    ]

    for epoch in range(
        1,
        max_epochs + 1,
    ):
        print(
            f"\n# Epoch {epoch}/{max_epochs}"
        )

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
            scaler=scaler,
            ema=ema,
            use_amp=use_amp,
            gradient_clip=cfg["training"][
                "gradient_clip"
            ],
            gradient_accumulation_steps=(
                cfg["training"][
                    "gradient_accumulation_steps"
                ]
            ),
        )

        val_metrics = evaluate_one_epoch(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
            use_amp=use_amp,
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"Train Loss: "
            f"{train_metrics['loss']:.4f}"
        )

        print(
            f"Train Acc:  "
            f"{train_metrics['accuracy']:.4f}"
        )

        print(
            f"Train Macro-F1: "
            f"{train_metrics['macro_f1']:.4f}"
        )

        print(
            f"Val Loss:   "
            f"{val_metrics['loss']:.4f}"
        )

        print(
            f"Val Acc:    "
            f"{val_metrics['accuracy']:.4f}"
        )

        print(
            f"Val Macro-F1: "
            f"{val_metrics['macro_f1']:.4f}"
        )

        print(
            f"LR: {current_lr:.8f}"
        )

        append_epoch_log(
            path=output_dir / "history.csv",
            epoch=epoch,
            lr=current_lr,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )

        val_accuracy = (
            val_metrics["accuracy"]
        )

        # EmoNeXt source selects by validation accuracy.
        if (
            val_accuracy
            > best_val_accuracy
        ):
            best_val_accuracy = (
                val_accuracy
            )

            early_stop_counter = 0

            save_checkpoint(
                path=(
                    checkpoint_dir
                    / "best.pt"
                ),
                model=model,
                optimizer=optimizer,
                ema=ema,
                scaler=scaler,
                scheduler=scheduler,
                best_metric=(
                    best_val_accuracy
                ),
                epoch=epoch,
                config=cfg,
            )

            print(
                "Saved new best checkpoint "
                f"(Val Acc="
                f"{best_val_accuracy:.4f})"
            )

        else:
            early_stop_counter += 1

            if (
                early_stop_counter
                >= patience
            ):
                print(
                    "Validation accuracy "
                    "did not improve for "
                    f"{patience} epochs."
                )

                print(
                    "Early stopping."
                )

                break

    # IMPORTANT:
    # Match EmoNeXt source behavior.
    # Do NOT reload best checkpoint here.
    # Test current EMA model directly.
    test_metrics = evaluate_test_ema(
        ema=ema,
        dataloader=test_loader,
        device=device,
        use_amp=use_amp,
    )

    print("\n# Test")

    print(
        f"Test Accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Test Macro-F1: "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(
        f"Test Balanced Accuracy: "
        f"{test_metrics['balanced_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()