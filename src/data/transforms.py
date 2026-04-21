from torchvision import transforms
import torch
import config
from PIL import Image


class Letterbox:
    """Resize while keeping aspect ratio, then center-pad to square."""
    def __init__(self, target_size: int = config.IMG_SIZE):
        self.target_size = target_size

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        ratio = self.target_size / max(img.size)
        new_size = tuple(round(x * ratio) for x in img.size)
        resized = img.resize(new_size, Image.Resampling.BILINEAR)

        canvas = Image.new("RGB", (self.target_size, self.target_size), (0, 0, 0))
        delta_w = (self.target_size - new_size[0]) // 2
        delta_h = (self.target_size - new_size[1]) // 2
        canvas.paste(resized, (delta_w, delta_h))
        return canvas


def get_train_transform() -> transforms.Compose:
    return transforms.Compose([
        Letterbox(target_size=config.IMG_SIZE),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=3),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
    ])


def get_eval_transform() -> transforms.Compose:
    return transforms.Compose([
        Letterbox(target_size=config.IMG_SIZE),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
    ])


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(config.IMG_MEAN).view(-1, 1, 1)
    std = torch.tensor(config.IMG_STD).view(-1, 1, 1)
    return tensor * std + mean