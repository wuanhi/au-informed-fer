import torch
from tqdm import tqdm

from evaluation.metrics import compute_metrics


def train_one_epoch(
    model,
    dataloader,
    loss_fn,
    optimizer,
    device,
    scaler,
    use_amp=True
):
    model.train()

    running_loss = 0.0
    all_targets = []
    all_predictions = []

    amp_enabled = use_amp and device.type == "cuda"

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        leave=False
    )

    for images, targets in progress_bar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = loss_fn(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        predictions = logits.argmax(dim=1)

        all_targets.extend(
            targets.detach().cpu().tolist()
        )

        all_predictions.extend(
            predictions.detach().cpu().tolist()
        )

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = running_loss / len(dataloader.dataset)

    metrics = compute_metrics(
        all_targets,
        all_predictions
    )

    metrics["loss"] = epoch_loss

    return metrics


@torch.no_grad()
def evaluate_one_epoch(
    model,
    dataloader,
    loss_fn,
    device,
    use_amp=True
):
    model.eval()

    running_loss = 0.0
    all_targets = []
    all_predictions = []

    amp_enabled = use_amp and device.type == "cuda"

    progress_bar = tqdm(
        dataloader,
        desc="Validation",
        leave=False
    )

    for images, targets in progress_bar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(images)
            loss = loss_fn(logits, targets)

        running_loss += loss.item() * images.size(0)

        predictions = logits.argmax(dim=1)

        all_targets.extend(
            targets.cpu().tolist()
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )

    epoch_loss = running_loss / len(dataloader.dataset)

    metrics = compute_metrics(
        all_targets,
        all_predictions
    )

    metrics["loss"] = epoch_loss

    return metrics