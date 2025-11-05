import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Configuration
    API_TITLE = "Minangkabau Food Classification API"
    API_DESCRIPTION = "API untuk klasifikasi makanan Minangkabau menggunakan MobileNetV3"
    API_VERSION = "1.0.0"
    
    # Security
    API_KEY = os.getenv("API_KEY", "your-default-api-key-here")
    API_KEY_NAME = "X-API-Key"
    
    # Model Configuration
    MODEL_PATH = os.getenv("MODEL_PATH", "models/your_model.h5")
    IMAGE_SIZE = (224, 224)
    
    # Class labels
    CLASS_NAMES = [
        "ayam_goreng",
        "ayam_pop", 
        "daging_rendang",
        "dendeng_batokok",
        "gulai_ikan",
        "gulai_tambusu",
        "gulai_tunjang",
        "telur_balado",
        "telur_dadar"
    ]

settings = Settings()