import torch.nn as nn
import torchvision


def set_parameter_requires_grad(model, feature_extracting: bool = True):
    """
    Freeze/unfreeze the model weights.

    Args:
        feature_extracting (bool): Weights are frozen if freature_extracting
            is True, and not frozen otherwise.

    Returns:
        None
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def create_model(
    num_classes: int = 3, model_name: str = "resnet50", model_weights: str = "DEFAULT"
):
    """
    Instantializes a pre-trained neural network with a custom output layer.

    Args:
        num_classes (int): The number of output classes for the image model.
        model_name (str): The name of the pre-trained model to use as a base. Defaults to "resnet50".
        model_weights (str): The weights to use for the pre-trained model. Defaults to "DEFAULT".

    Return:
        model (): A pre-trained neural network model with a custom output layer.
    """
    model = torchvision.models.get_model(model_name, weights=model_weights)
    set_parameter_requires_grad(model, feature_extracting=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model
