import json
import logging

import requests
import streamlit as st
from general_utils import setup_logging
from PIL import Image


def main():
    """
    This main function does the following:
        - Loads logging config
        - Sends a post request to the API endpoint /xnn/predict/
        - Retrieves inference results from the API endpoint /xnn/predict/
        - Outputs prediction results on the dashboard
    """

    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    setup_logging("conf/base/logging.yaml")

    # Loads dashboard
    logger.info("Loading dashboard...")
    st.title("Explainable Neural Network")
    st.caption(
        "This is a user interface to classify chest X-ray image (normal vs viral pneumonia vs bacterial pneumonia).\
               A grad-CAM heatmap will show parts of an image that the image model emphasizes on for classification."
    )

    # Uploads image to predict
    upload = st.file_uploader("Upload image:")

    # Retrieves prediction if an image is uploaded
    if st.button("Get classification"):
        if upload is not None:
            logger.info("Conducting inferencing on image input...")

            with st.spinner("Predicting..."):
                # Sends post request to API endpoint /xnn/predict/
                address = "http://localhost:8000/xnn/predict"
                headers = {"accept": "application/json"}
                files = {"file": (upload.name, upload.getvalue(), "image/jpeg")}
                r = requests.post(address, headers=headers, files=files)

            if r.status_code == 200:
                # Retrieve prediction results
                res_dict = json.loads(r.text)
                pred = res_dict["prediction"]
                logger.info(f"Inferencing has completed. Prediction is {pred}")

                col1, col2 = st.columns(2)
                # Display prediction results
                with col1:
                    image_input = Image.open(upload)
                    st.image(image_input)
                    st.write(f"Image is predicted as {pred}")
                # Display gradCAM results
                # with col2:
                # st.write("These are the regions emphasized by the image model.")

        # If user clicks button to get prediction but did not upload image
        else:
            st.write("No image uploaded. Please upload an image.")


if __name__ == "__main__":
    main()
