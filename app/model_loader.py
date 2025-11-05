import os
from dotenv import load_dotenv
import tensorflow as tf

load_dotenv()

def load_model():
    model_path = os.getenv("MODEL_PATH")
    model = tf.keras.models.load_model(model_path)
    return model
