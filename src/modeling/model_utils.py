import torch.nn as nn
import torchvision


def set_parameter_requires_grad(model, feature_extracting=True):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def create_model(
    num_classes: int = 2, model_name: str = "resnet50", model_weights: str = "DEFAULT"
):
    model = torchvision.models.get_model(model_name, weights=model_weights)
    set_parameter_requires_grad(model, feature_extracting=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model
