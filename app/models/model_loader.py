import tensorflow as tf
import numpy as np
from PIL import Image
import io
from app.config import settings

class MinangFoodClassifier:
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = tf.keras.models.load_model(settings.MODEL_PATH)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    
    def preprocess_image(self, image_bytes):
        """Preprocess image for model prediction"""
        try:
            # Open image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image
            image = image.resize(settings.IMAGE_SIZE)
            
            # Convert to array and normalize
            image_array = np.array(image) / 255.0
            
            # Add batch dimension
            image_batch = np.expand_dims(image_array, axis=0)
            
            return image_batch
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            raise e
    
    def predict(self, image_bytes):
        """Make prediction on image"""
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_bytes)
            
            # Make prediction
            predictions = self.model.predict(processed_image)
            
            # Get top prediction
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # Get class name
            predicted_class = settings.CLASS_NAMES[predicted_class_idx]
            
            # Get top 3 predictions
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            top_3_predictions = [
                {
                    "class": settings.CLASS_NAMES[i],
                    "confidence": float(predictions[0][i])
                }
                for i in top_3_indices
            ]
            
            return {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "all_predictions": top_3_predictions
            }
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            raise e

# Global model instance
classifier = MinangFoodClassifier()