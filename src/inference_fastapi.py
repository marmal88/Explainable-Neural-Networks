import io
import logging

import torch
import uvicorn
from fastapi import FastAPI, UploadFile
from hydra import compose, initialize
from PIL import Image

from .general_utils import setup_logging
from .modeling.model_utils import create_model, load_model
from .modeling.models import ImageClassifier
from .modeling.preprocess import ImageTransforms

# Set up logging
logger = logging.getLogger(__name__)
logger.info("Setting up logging configuration.")
setup_logging(logging_config_path="conf/base/logging.yaml")

# Initialize config
with initialize(
    # version_base=None,
    config_path="../conf/base"
):
    cfg = compose("pipelines.yaml")

app = FastAPI()


@app.on_event("startup")
def startup_event():
    """
    Startup event trigger
    """
    logger.info("Starting server...")

    # Load image model
    logger.info("Loading model weights...")
    model = ImageClassifier(
        backbone=create_model(
            num_classes=int(cfg.dataset.num_classes),
            model_name=str(cfg.model.model_name),
        ),
        learning_rate=cfg.train.params.learning_rate,
    )
    app.model = load_model(model, cfg.inference.model_path)

    # Load image transforms
    logger.info("Loading image transforms...")
    app.transform_img = ImageTransforms(
        int(cfg.train.transforms.image_size)
    ).test_transforms()
    logger.info("Server startup completed.")


@app.post("/xnn/predict")
async def predict(file: UploadFile):
    """
    Predicts the class of an image.

    Args:
        file (UploadFile): Uploaded image

    Returns:
        response_payload (Dict): Prediction of the uploaded image
    """
    # Read image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))

    # Transform image
    img = app.transform_img(img)
    img = img.resize(
        1, 3, cfg.train.transforms.image_size, cfg.train.transforms.image_size
    )

    # Predict class of image
    app.model.eval()
    with torch.no_grad():
        logits = app.model(img)
    pred_ind = torch.argmax(logits, dim=1)

    # TODO: manually code first, figure out where to put this later
    labels = ["bacteria", "normal", "virus"]

    response_payload = {"prediction": labels[pred_ind]}
    return response_payload


if __name__ == "__main__":
    uvicorn.run(
        "src.inference_fastapi:app", host="0.0.0.0", reload=True, log_level="info"
    )
