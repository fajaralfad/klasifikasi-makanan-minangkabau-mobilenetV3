from PIL import Image
import numpy as np

# Label kelas sesuai dataset
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

IMAGE_SIZE = 224  # sesuai model

def preprocess(image_file):
    img = Image.open(image_file).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img, dtype="float32") / 255.0      
    img = np.expand_dims(img, axis=0)               
    return img
