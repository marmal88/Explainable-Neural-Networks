"""This package contains functions to instantialize a pre-trained neural network
from torchvision for image classification."""

import functools

import torch
import torchvision
from torch import nn


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


def get_layers(model: torch.nn.Module):
    """get_layers"""
    children = list(model.children())
    return (
        [model]
        if len(children) == 0
        else [ci for c in children for ci in get_layers(c)]
    )


def rgetattr(obj, attr, *args):
    """rgetattr"""

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def rsetattr(obj, attr, val):
    """set and get attribute dynamically"""
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def create_model(
    num_classes: int = 3, model_name: str = "resnet50", model_weights: str = "DEFAULT"
):
    """
    Instantializes a pre-trained neural network with a custom output layer.

    Args:
        num_classes (int): The number of output classes for the image model.
        model_name (str): The name of the pre-trained model to use as a base.
            Defaults to "resnet50".
        model_weights (str): The weights to use for the pre-trained model.
            Defaults to "DEFAULT".

    Return:
        model (): A pre-trained neural network model with a custom output layer.
    """
    model = torchvision.models.get_model(model_name, weights=model_weights)

    last_layer_name = [
        name for name, _ in model.named_modules() if isinstance(model, torch.nn.Module)
    ]

    set_parameter_requires_grad(model, feature_extracting=True)
    final_layer = get_layers(model)[-1]
    num_ftrs = final_layer.in_features
    head = nn.Linear(num_ftrs, num_classes)
    rsetattr(model, last_layer_name[-1], head)
    return model
