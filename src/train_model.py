"""This package runs the model training."""

import logging
import os

import hydra
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig

from .modeling.data_loaders import ImageClassificationDataModule
from .modeling.model_utils import create_model
from .modeling.models import ImageClassifier
from .modeling.preprocess import ImageTransforms


# pylint: disable=logging-fstring-interpolation
@hydra.main(version_base=None, config_path="../conf/base", config_name="pipelines.yaml")
def run(cfg: DictConfig) -> None:
    """
    This function runs the model training:
      1) Creates image transformation
      2) Creates the model and trainer
      3) Prepares dataset and dataloader
      4) Fits the model
      5) Evaluates the model

    Args:
        cfg (DictConfig): configuration from pipelines.yaml

    Returns:
        None
    """
    logging.info("Instantiating image transformation")
    train_transform_img = ImageTransforms(
        cfg.train.transforms.image_size
    ).train_transforms()
    test_transform_img = ImageTransforms(
        cfg.train.transforms.image_size
    ).test_transforms()

    trainer = Trainer(
        max_epochs=cfg.train.params.epochs,
        logger=CSVLogger(save_dir="logs/", name="xnn"),
    )

    logging.info("Instantiating model")
    model = ImageClassifier(
        backbone=create_model(), learning_rate=cfg.train.params.learning_rate
    )

    train_data_path = os.path.join(
        hydra.utils.get_original_cwd(), cfg.data_paths.train_data_path
    )
    val_data_path = os.path.join(
        hydra.utils.get_original_cwd(), cfg.data_paths.val_data_path
    )
    test_data_path = os.path.join(
        hydra.utils.get_original_cwd(), cfg.data_paths.test_data_path
    )
    meta_data_path = os.path.join(
        hydra.utils.get_original_cwd(), cfg.data_paths.meta_data_path
    )

    logging.info("Instantiating data module")
    data_module = ImageClassificationDataModule(
        train_image_folder=train_data_path,
        val_image_folder=val_data_path,
        test_image_folder=test_data_path,
        meta_data_path=meta_data_path,
        train_transform_img=train_transform_img,
        test_transform_img=test_transform_img,
        batch_size=cfg.train.params.batch_size,
    )

    logging.info("Training model")
    trainer.fit(model, datamodule=data_module)

    train_acc = trainer.test(dataloaders=data_module.train_dataloader())[0][
        "test_accuracy"
    ]
    val_acc = trainer.test(dataloaders=data_module.val_dataloader())[0]["test_accuracy"]
    test_acc = trainer.test(dataloaders=data_module.test_dataloader())[0][
        "test_accuracy"
    ]

    logging.info(
        f"Train Acc {train_acc*100:.2f}%"
        f" | Val Acc {val_acc*100:.2f}%"
        f" | Test Acc {test_acc*100:.2f}%"
    )


if __name__ == "__main__":
    run()
