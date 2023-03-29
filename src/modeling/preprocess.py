import torchvision.transforms as T


class ImageTransforms:
    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    def train_transforms(self):
        transform = T.Compose(
            [
                T.Resize(self.image_size),
                T.CenterCrop(self.image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return transform

    def test_transforms(self):
        transform = T.Compose(
            [
                T.Resize(self.image_size),
                T.CenterCrop(self.image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return transform
