from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn
import os


from app.config import settings
from app.auth import get_api_key
from app.models.model_loader import classifier
from app.utils import validate_image

app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Minangkabau Food Classification API",
        "version": settings.API_VERSION,
        "available_classes": settings.CLASS_NAMES
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": classifier.model is not None}

@app.get("/classes")
async def get_classes(api_key: str = Depends(get_api_key)):
    """Get all available food classes"""
    return {
        "classes": [
            {"id": i, "name": name} 
            for i, name in enumerate(settings.CLASS_NAMES)
        ]
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    api_key: str = Depends(get_api_key)
):
    """
    Predict Minangkabau food from image
    
    - **file**: Image file (JPEG, PNG, JPG, WEBP) max 10MB
    """
    try:
        # Validate image
        validate_image(file)
        
        # Read file content
        contents = await file.read()
        
        # Make prediction
        result = classifier.predict(contents)
        
        return {
            "success": True,
            "prediction": result,
            "filename": file.filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/batch-predict")
async def batch_predict(
    files: List[UploadFile] = File(...),
    api_key: str = Depends(get_api_key)
):
    """
    Batch prediction for multiple images
    """
    try:
        results = []
        
        for file in files:
            try:
                validate_image(file)
                contents = await file.read()
                result = classifier.predict(contents)
                results.append({
                    "filename": file.filename,
                    "success": True,
                    "prediction": result
                })
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "batch_results": results,
            "total_files": len(files),
            "successful_predictions": len([r for r in results if r["success"]])
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Railway pakai $PORT
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)