"""This package contains classes for image augmentation."""

from typing import Callable

import torchvision.transforms as T


class ImageTransforms:
    """
    Initialise an object of class ImageTransforms.

    Args:
        image_size (int): size to crop image to.

    Returns:
        None
    """

    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    def train_transforms(self) -> Callable:
        """
        Creates image transformation function.

        Args:
            None

        Returns:
            transform (Callable): function to process image for train data.
        """
        transform = T.Compose(
            [
                T.Grayscale(3),
                T.RandomRotation(degrees=10),
                T.RandomResizedCrop(self.image_size, scale=(0.9, 1.05), ratio=(1, 1)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return transform

    def test_transforms(self) -> Callable:
        """
        Creates image transformation function.

        Args:
            None

        Returns:
            transform (Callable): function to process image for validation/test data.
        """
        transform = T.Compose(
            [
                T.Grayscale(3),
                T.Resize(self.image_size),
                T.CenterCrop(self.image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return transform
