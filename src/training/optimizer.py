import torch


def build_optimizer(model, cfg):
    name = cfg["optimizer"]["name"].lower()
    lr = cfg["optimizer"]["lr"]
    weight_decay = cfg["optimizer"]["weight_decay"]

    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    raise ValueError(f"Unsupported optimizer: {name}")