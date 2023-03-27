import logging
import os

import hydra
from lightning.pytorch import Trainer
from omegaconf import DictConfig

from .modeling.data_loaders import ImageClassificationDataModule
from .modeling.model_utils import create_model
from .modeling.models import ImageClassifier
from .modeling.preprocess import ImageTransforms


@hydra.main(config_path="../../conf/base", config_name="pipelines.yaml")
def run(cfg: DictConfig) -> None:
    logging.info(os.getcwd())

    train_transform_img = ImageTransforms(
        cfg.train.transforms.image_size
    ).train_transforms()
    test_transform_img = ImageTransforms(
        cfg.train.transforms.image_size
    ).test_transforms()
    # logging.info("made transforms")
    # logging.info(hydra.utils.get_original_cwd())

    trainer = Trainer(max_epochs=cfg.train.params.epochs)
    model = ImageClassifier(
        backbone=create_model(), learning_rate=cfg.train.params.learning_rate
    )
    # logging.info(os.getcwd())
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
    data_module = ImageClassificationDataModule(
        train_image_folder=train_data_path,
        val_image_folder=val_data_path,
        test_image_folder=test_data_path,
        meta_data_path=meta_data_path,
        train_transform_img=train_transform_img,
        test_transform_img=test_transform_img,
        batch_size=cfg.train.params.batch_size,
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    run()
