from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import os
import base64
import cv2
import numpy as np
from ultralytics import YOLO
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Configuration
MODEL_PATH = 'model/runs/segment/train/weights/best.pt'  # Path to your model
TEMP_DIR = 'temp'  # Temporary directory for storing images

# Create temp directory if it doesn't exist
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize FastAPI
app = FastAPI()

# Enable CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper functions
def base64_to_image(base64_string):
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    img_bytes = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

def image_to_base64(image):
    _, buffer = cv2.imencode('.png', image)
    return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"

def process_image(base64_image):
    image = base64_to_image(base64_image)
    model = YOLO(MODEL_PATH)
    results = model(image)
    detection_image = results[0].plot()
    detection_image = cv2.cvtColor(detection_image, cv2.COLOR_RGB2BGR)

    green_pixel_count = 0
    non_green_pixel_count = 0
    green_only_image = None
    non_green_only_image = None

    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        highest_conf_index = np.argmax(scores)        
        highest_confidence_score = scores[highest_conf_index]
        highest_confidence_class = results[0].names[classes[highest_conf_index]]
        
        results[0].masks.data = results[0].masks.data[highest_conf_index:highest_conf_index + 1]
        results[0].boxes.data = results[0].boxes.data[highest_conf_index:highest_conf_index + 1]
        mask = masks[highest_conf_index]
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        segmented_image = (image * mask_3d).astype(np.uint8)
        lower_green = np.array([33, 0, 0])
        upper_green = np.array([85, 255, 255])
        hsv = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2HSV)
        non_white_and_black_mask = ~((segmented_image == [255, 255, 255]).all(axis=2) | (segmented_image == [0, 0, 0]).all(axis=2))
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_pixel_count = np.count_nonzero(green_mask & non_white_and_black_mask)
        green_only_image = cv2.bitwise_and(segmented_image, segmented_image, mask=green_mask)
        non_green_mask = cv2.bitwise_not(green_mask)
        non_green_pixel_count = np.count_nonzero(non_green_mask & non_white_and_black_mask)
        non_green_only_image = cv2.bitwise_and(segmented_image, segmented_image, mask=non_green_mask)

    return {
        'classname': highest_confidence_class,
        'confidence': float("{:.2f}".format(highest_confidence_score*100)),
        'green_only_image':      image_to_base64(green_only_image)       if green_only_image is not None else None,
        'non_green_only_image':  image_to_base64(non_green_only_image)   if non_green_only_image is not None else None,
        'detection_image':       image_to_base64(detection_image)        if detection_image is not None else None,
        'segmented_image':       image_to_base64(segmented_image)        if green_only_image is not None else None,
        'green_pixel_count':     green_pixel_count,
        'non_green_pixel_count': non_green_pixel_count,
        'severity': float("{:.2f}".format((non_green_pixel_count / (green_pixel_count + non_green_pixel_count)) * 100)) if (green_pixel_count + non_green_pixel_count) > 0 else 0
    }

# Request model
class ImageRequest(BaseModel):
    image: str

# Routes
@app.post("/predict")
async def predict(request: ImageRequest):
    try:
        results = process_image(request.image)
        return JSONResponse(content=results)
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = 5001
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
