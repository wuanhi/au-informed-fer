import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchvision.utils import make_grid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataloader import build_dataloaders
from src.models.convnext_stn import build_stn_convnext_tiny
from src.training.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.seed import set_seed


def unnormalize_image(x):
    return x.clamp(0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="outputs/stn_visualization")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = load_config(args.data_config)
    set_seed(cfg["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, val_loader, _ = build_dataloaders(data_cfg, cfg)

    model = build_stn_convnext_tiny(
        num_classes=cfg["model"]["num_classes"],
        pretrained=False,
        drop_path_rate=cfg["model"]["drop_path_rate"],
    ).to(device)

    load_checkpoint(
        path=args.checkpoint,
        model=model,
        device=device,
    )

    model.eval()

    inputs_list = []
    transformed_list = []
    theta_rows = []

    seen = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            transformed, theta = model.forward_stn_with_theta(images)

            batch_size = images.size(0)

            for i in range(batch_size):
                if seen >= args.num_samples:
                    break

                inputs_list.append(images[i].cpu())
                transformed_list.append(transformed[i].cpu())

                t = theta[i].cpu().numpy()

                theta_rows.append(
                    {
                        "sample_id": seen,
                        "label": int(labels[i]),
                        "a": float(t[0, 0]),
                        "b": float(t[0, 1]),
                        "tx": float(t[0, 2]),
                        "c": float(t[1, 0]),
                        "d": float(t[1, 1]),
                        "ty": float(t[1, 2]),
                    }
                )

                seen += 1

            if seen >= args.num_samples:
                break

    inputs = torch.stack(inputs_list)
    transformed = torch.stack(transformed_list)

    diff = torch.abs(inputs - transformed)

    pair_images = []
    for i in range(args.num_samples):
        pair_images.append(unnormalize_image(inputs[i]))
        pair_images.append(unnormalize_image(transformed[i]))
        pair_images.append(unnormalize_image(diff[i] * 5.0))

    grid = make_grid(
        pair_images,
        nrow=3,
        padding=4,
    )

    plt.figure(figsize=(8, args.num_samples * 2))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.title("STN visualization: input | transformed | abs diff x5")
    plt.tight_layout()
    plt.savefig(output_dir / "stn_input_vs_transformed.png", dpi=200)
    plt.close()

    pd.DataFrame(theta_rows).to_csv(
        output_dir / "stn_theta.csv",
        index=False,
    )

    print(f"Saved image: {output_dir / 'stn_input_vs_transformed.png'}")
    print(f"Saved theta: {output_dir / 'stn_theta.csv'}")


if __name__ == "__main__":
    main()