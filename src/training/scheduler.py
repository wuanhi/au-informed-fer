import torch


def build_scheduler(optimizer, cfg):
    scheduler_cfg = cfg["scheduler"]
    name = scheduler_cfg["name"].lower()

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg["training"]["epochs"]
        )

    raise ValueError(f"Unsupported scheduler: {name}")