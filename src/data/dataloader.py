from torch.utils.data import DataLoader

from data.fer2013_dataset import FER2013Dataset
from data.transforms import get_train_transform, get_val_transform


def build_dataloaders(data_cfg, experiment_cfg):
    csv_path = data_cfg["dataset"]["csv_path"]

    image_size = data_cfg["image"]["size"]

    batch_size = data_cfg["dataloader"]["batch_size"]
    num_workers = data_cfg["dataloader"]["num_workers"]

    train_split = data_cfg["splits"]["train"]
    val_split = data_cfg["splits"]["val"]
    test_split = data_cfg["splits"]["test"]

    augmentation_cfg = experiment_cfg["augmentation"]

    train_dataset = FER2013Dataset(
        csv_path=csv_path,
        split=train_split,
        transform=get_train_transform(
            image_size=image_size,
            augmentation_cfg=augmentation_cfg
        )
    )

    val_dataset = FER2013Dataset(
        csv_path=csv_path,
        split=val_split,
        transform=get_val_transform(image_size)
    )

    test_dataset = FER2013Dataset(
        csv_path=csv_path,
        split=test_split,
        transform=get_val_transform(image_size)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader