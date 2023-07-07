from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import cv2
import numpy as np
# from PIL import Image
import base64, os, io, json
import time

from matching import matching_operation

app = FastAPI(
    title="image_matching",
    description="return base64 result image after matching",
    version="0.1.0")
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)

class ImageRequest(BaseModel):
    image1: str
    image2: str

class ImageResponse(BaseModel):
    output_image: str

def check_and_modify_b64(base64_image):
  if ";base64," in base64_image:
      base64_image = base64_image.split(";base64,")[1].encode("utf8")
  else:
      base64_image = base64_image.encode("utf8")
  return base64_image

def process_images(image1_base64: str, image2_base64: str) -> str:
    start = time.time()
    image1_base64 = check_and_modify_b64(image1_base64)
    image2_base64 = check_and_modify_b64(image2_base64)

    # try:
    #     imgdata = base64.b64decode(image1_base64)
    #     image = Image.open(io.BytesIO(imgdata))
    #     image.save('image1.png')
    #     imgdata = base64.b64decode(image2_base64)
    #     image = Image.open(io.BytesIO(imgdata))
    #     image.save('image2.png')
    # except Exception as e:
    #         print("Exception in image save", e)

    # Convert Base64 strings to NumPy arrays
    image1_data = base64.b64decode(image1_base64)
    image2_data = base64.b64decode(image2_base64)
    nparr1 = np.frombuffer(image1_data, np.uint8)
    nparr2 = np.frombuffer(image2_data, np.uint8)

    # Decode the images using OpenCV
    image1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
    image2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)

    # Perform your desired image processing here
    # For example, concatenate the images side by side
    combined_image = matching_operation(image1, image2)

    # Encode the output image to Base64
    _, output_image_data = cv2.imencode('.jpg', combined_image)
    output_image_base64 = base64.b64encode(output_image_data).decode('utf-8')
    end = time.time()
    print("Total time taken =", end-start)
    return output_image_base64

@app.post("/process_images")
def process_images_endpoint(image_request: ImageRequest) -> ImageResponse:
    try:
        output_image_base64 = process_images(image_request.image1, image_request.image2)
    except Exception as e:
        print('Exception', e)
    return ImageResponse(output_image=output_image_base64)

PORT = int(os.getenv("PORT")) if os.getenv("PORT") else 8000
uvicorn.run(app, host="0.0.0.0", port=PORT)