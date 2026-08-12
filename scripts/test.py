import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from ema_pytorch import EMA

from src.data.dataloader import build_dataloaders
from src.models.convnext_backbone import build_convnext_tiny
from src.training.checkpoint import load_checkpoint
from src.training.trainer import evaluate_test_ema
from src.utils.config import load_config
from src.utils.seed import set_seed


def build_model(model_cfg):
    if model_cfg["name"] == "convnext_tiny":
        return build_convnext_tiny(
            num_classes=model_cfg["num_classes"],
            pretrained=False,
            drop_path_rate=model_cfg["drop_path_rate"],
        )

    raise ValueError(
        f"Unsupported model: {model_cfg['name']}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--data-config",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = load_config(args.data_config)

    # Match reproduction seed/config.
    set_seed(cfg["training"]["seed"])

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # build_dataloaders() now creates:
    # train -> regular augmentation
    # val   -> RandomCrop
    # test  -> TenCrop
    _, _, test_loader = build_dataloaders(
        data_cfg,
        cfg,
    )

    print(
        f"Test samples: {len(test_loader.dataset)}"
    )

    # IMPORTANT:
    # pretrained=False here because checkpoint already
    # contains the FER2013 fine-tuned model.
    model = build_model(
        cfg["model"]
    ).to(device)

    # EmoNeXt test protocol evaluates EMA weights.
    ema = EMA(
        model,
        beta=cfg["ema"]["beta"],
        update_every=cfg["ema"]["update_every"],
    ).to(device)

    checkpoint = load_checkpoint(
        path=args.checkpoint,
        model=model,
        ema=ema,
        device=device,
    )

    print(
        f"Loaded checkpoint: {args.checkpoint}"
    )

    if "epoch" in checkpoint:
        print(
            f"Checkpoint epoch: {checkpoint['epoch']}"
        )

    if "best_acc" in checkpoint:
        print(
            "Best validation accuracy: "
            f"{checkpoint['best_acc']:.4f}"
        )

    # EmoNeXt protocol:
    #
    # [B, 10, 3, 224, 224]
    #       ↓
    # EMA model
    #       ↓
    # [B, 10, 7] logits
    #       ↓
    # mean over 10 crops
    #       ↓
    # prediction
    metrics = evaluate_test_ema(
        ema=ema,
        dataloader=test_loader,
        device=device,
        use_amp=cfg["training"]["amp"],
    )

    print()
    print("Test results")
    print(
        f"Accuracy:          "
        f"{metrics['accuracy']:.4f}"
    )
    print(
        f"Macro-F1:          "
        f"{metrics['macro_f1']:.4f}"
    )
    print(
        f"Balanced Accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print()
    print("Per-class metrics")

    for name, values in metrics["per_class"].items():
        print(
            f"{name:8s}: "
            f"P={values['precision']:.4f} "
            f"R={values['recall']:.4f} "
            f"F1={values['f1']:.4f} "
            f"(support={values['support']})"
        )


if __name__ == "__main__":
    main()