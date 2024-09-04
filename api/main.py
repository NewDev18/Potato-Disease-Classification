from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../final_model.h5")

class_name = ["Early_blight", "Late_blight", "Healthy"]

Allowed_Extensions = {"image/jpeg", "image/png"}

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))

    pass
    return image

@app.post("/predict")
async def predict(file:UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_array = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_array)
    print(predictions)
    predicted_class = class_name[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)