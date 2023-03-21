import hydra
from omegaconf import DictConfig
from lightning.pytorch import Trainer

from .modeling.preprocess import ImageTransforms
from .modeling.models import ImageClassifier
from .modeling.model_utils import create_model
from .modeling.data_loaders import ImageClassificationDataModule 

@hydra.main(config_path="conf/base",
            config_name="pipelines.yaml")
def run(cfg: DictConfig) -> None:
    train_transform_img = ImageTransforms(cfg.train.transforms.image_size).train_transforms
    test_transform_img = ImageTransforms(cfg.train.transforms.image_size).test_transforms

    trainer = Trainer(
        max_epochs = cfg.train.params.epochs
    )
    model = ImageClassifier(backbone=create_model(), learning_rate = cfg.params.learning_rate)
    data_module = ImageClassificationDataModule(
          train_transform_img=train_transform_img,
          test_transform_img=test_transform_img,
          batch_size=cfg.train.params.batch_size,
          train_data_path=cfg.data_paths.train_data_path,
          validation_data_path=cfg.data_paths.validation_data_path,
          test_data_path=cfg.data_paths.test_data_path)
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    run()