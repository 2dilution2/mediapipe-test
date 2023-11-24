from typing import Annotated, Optional

from fastapi import FastAPI, File, UploadFile, Form

from utils import visualize

# STEP 1: Import the necessary modules.
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an ObjectDetector object.
model_path = 'efficientdet_lite0.tflite'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

app = FastAPI()

@app.post("/files/")
async def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file)}

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    contents = await file.read()
    return {"filename": file.filename, "file_size":len(contents)}

import io
from PIL import Image

@app.post("/predict")
async def predict_api(image_file: UploadFile):

    # 0. read bytes from http
    contents = await image_file.read()
    
    # 1. make buffer from bytes
    buffer = io.BytesIO(contents)

    # 2. decode image from buffer
    pil_img = Image.open(buffer)

    # STEP 3: Load the input image.
    # image = mp.Image.create_from_file(IMAGE_FILE)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))

    # STEP 4: Detect objects in the input image.
    detection_result = detector.detect(image)

    return{'result' : detection_result}

    # STEP 5: Process the detection result. In this case, visualize it.
    # image_copy = np.copy(image.numpy_view())
    # annotated_image = visualize(image_copy, detection_result)
    # rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

from fastapi.responses import StreamingResponse
import cv2

@app.post("/predict_img")
async def predict_api_img(image_file: UploadFile):

    # 0. read bytes from http
    contents = await image_file.read()
    
    # 1. make buffer from bytes
    buffer = io.BytesIO(contents)

    # 2. decode image from buffer
    pil_img = Image.open(buffer)

    # STEP 3: Load the input image.
    # image = mp.Image.create_from_file(IMAGE_FILE)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))

    # STEP 4: Detect objects in the input image.
    detection_result = detector.detect(image)

    # STEP 5: Process the detection result. In this case, visualize it.
    image_copy = np.copy(image.numpy_view())
    annotated_image = visualize(image_copy, detection_result)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # STEP 6: encode
    img_encode = cv2.imencode('.png', rgb_annotated_image)[1]
    image_stream = io.BytesIO(img_encode.tobytes())
    return StreamingResponse(image_stream, media_type="image/png")