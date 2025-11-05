from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn
import os
import logging

from app.config import settings
from app.auth import get_api_key
from app.models.model_loader import classifier
from app.utils import validate_image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

@app.on_event("startup")
async def startup_event():
    """Startup event - log startup completion"""
    logger.info("="*50)
    logger.info("Application starting up...")
    logger.info(f"API Version: {settings.API_VERSION}")
    logger.info(f"Model will be loaded on first prediction request (lazy loading)")
    logger.info("Application startup complete!")
    logger.info("="*50)

@app.get("/")
async def root():
    return {
        "message": "Minangkabau Food Classification API",
        "version": settings.API_VERSION,
        "status": "running",
        "model_status": "ready" if classifier.is_loaded else "will load on first request"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint - responds quickly without loading model"""
    return {
        "status": "healthy",
        "api_version": settings.API_VERSION,
        "model_loaded": classifier.is_loaded,
        "model_status": "loaded" if classifier.is_loaded else "ready to load"
    }

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
        logger.info(f"Received prediction request for file: {file.filename}")
        
        # Validate image
        validate_image(file)
        
        # Read file content
        contents = await file.read()
        logger.info(f"File size: {len(contents)} bytes")
        
        # Make prediction (model will auto-load if not loaded)
        result = classifier.predict(contents)
        
        if not result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Prediction failed")
            )
        
        logger.info(f"Prediction successful: {result.get('predicted_class')}")
        
        return {
            "success": True,
            "prediction": result,
            "filename": file.filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
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
        logger.info(f"Received batch prediction request for {len(files)} files")
        results = []
        
        for file in files:
            try:
                validate_image(file)
                contents = await file.read()
                result = classifier.predict(contents)
                
                if result.get("success"):
                    results.append({
                        "filename": file.filename,
                        "success": True,
                        "prediction": result
                    })
                else:
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "error": result.get("error", "Unknown error")
                    })
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}")
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": str(e)
                })
        
        successful = len([r for r in results if r["success"]])
        logger.info(f"Batch prediction complete: {successful}/{len(files)} successful")
        
        return {
            "batch_results": results,
            "total_files": len(files),
            "successful_predictions": successful
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )