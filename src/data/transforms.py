from torchvision import transforms
from torchvision.models import ConvNeXt_Tiny_Weights


def get_train_transform(image_size, augmentation_cfg):
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    mean = weights.transforms().mean
    std = weights.transforms().std

    transform_list = [
        transforms.Grayscale(num_output_channels=3)
    ]

    if augmentation_cfg.get("random_crop", False):
        scale = augmentation_cfg.get("crop_scale", [0.8, 1.0])

        transform_list.append(
            transforms.RandomResizedCrop(
                size=image_size,
                scale=tuple(scale)
            )
        )
    else:
        transform_list.append(
            transforms.Resize((image_size, image_size))
        )

    if augmentation_cfg.get("random_rotation", False):
        degrees = augmentation_cfg.get("rotation_degrees", 10)

        transform_list.append(
            transforms.RandomRotation(degrees=degrees)
        )

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return transforms.Compose(transform_list)


def get_val_transform(image_size):
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    mean = weights.transforms().mean
    std = weights.transforms().std

    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])