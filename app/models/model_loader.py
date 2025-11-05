import tensorflow as tf
import numpy as np
from PIL import Image
import io
import logging
from app.config import settings

logger = logging.getLogger(__name__)

class MinangFoodClassifier:
    def __init__(self):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.is_loaded = False
       
        logger.info("MinangFoodClassifier initialized (model will be loaded on first request)")
    
    def load_model(self):
        """Load TFLite model (lazy loading)"""
        if self.is_loaded:
            return  # Model sudah loaded, skip
            
        try:
            logger.info(f"Loading TFLite model from: {settings.MODEL_PATH}")
            
            # Load TFLite model
            self.interpreter = tf.lite.Interpreter(model_path=settings.MODEL_PATH)
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            self.is_loaded = True
            logger.info("TFLite model loaded successfully")
            logger.info(f"Input details: {self.input_details[0]['shape']}")
            logger.info(f"Output details: {self.output_details[0]['shape']}")
            
        except Exception as e:
            logger.error(f"Failed to load TFLite model: {e}")
            self.is_loaded = False
            raise e
    
    def preprocess_image(self, image_bytes):
        """Preprocess image for TFLite model"""
        try:
            # Open image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image sesuai input model
            input_shape = self.input_details[0]['shape']
            target_size = (input_shape[1], input_shape[2])  # (height, width)
            image = image.resize(target_size)
            
            # Convert to array dan normalize
            image_array = np.array(image, dtype=np.float32) / 255.0
            
            # Add batch dimension
            image_batch = np.expand_dims(image_array, axis=0)
            
            return image_batch
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise e
    
    def predict(self, image_bytes):
        """Make prediction menggunakan TFLite"""
        # Load model on first predict call (lazy loading)
        if not self.is_loaded:
            logger.info("Model not loaded yet, loading now...")
            try:
                self.load_model()
            except Exception as e:
                return {
                    "error": f"Failed to load model: {str(e)}",
                    "success": False
                }
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_bytes)
            
            # Set input tensor
            self.interpreter.set_tensor(
                self.input_details[0]['index'], 
                processed_image
            )
            
            # Run inference
            self.interpreter.invoke()
            
            # Get prediction results
            predictions = self.interpreter.get_tensor(
                self.output_details[0]['index']
            )
            
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
                "success": True,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "all_predictions": top_3_predictions
            }
            
        except Exception as e:
            logger.error(f"Error during TFLite prediction: {e}")
            return {
                "error": str(e),
                "success": False
            }

# Global TFLite model instance (tanpa load model)
classifier = MinangFoodClassifier()