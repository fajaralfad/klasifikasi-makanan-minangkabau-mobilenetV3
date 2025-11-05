from fastapi import FastAPI, UploadFile, File
from app.model_loader import load_model
from app.utils import preprocess, CLASS_NAMES
import numpy as np

app = FastAPI(title="API Klasifikasi Makanan Minangkabau")

model = load_model()

@app.get("/")
def home():
    return {"message": "API Klasifikasi Makanan Minangkabau Aktif ✅"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = preprocess(file.file)

    predictions = model.predict(img)[0]  # hasil softmax
    class_id = np.argmax(predictions)
    confidence = float(predictions[class_id])

    return {
        "class": CLASS_NAMES[class_id],
        "confidence": round(confidence, 4)
    }
