import os
import io
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

load_dotenv()

app = Flask(__name__)

API_KEY = os.getenv("API_KEY", "TEMPE12345")

# Simple dummy model class (used if no .pth provided)
class DummyModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(224*224*3, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'vit_tempe_model.pth')

# Try to load real model; if not present, use DummyModel with random weights
if os.path.exists(MODEL_PATH):
    try:
        model = torch.load(MODEL_PATH, map_location='cpu')
        print("Loaded model from", MODEL_PATH)
    except Exception as e:
        print("Failed loading model file, using DummyModel. Error:", e)
        model = DummyModel()
else:
    print("No model file found, using DummyModel.")
    model = DummyModel()

model.eval()

labels = ['mentah', 'setengah_matang', 'matang']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

@app.before_request
def check_api_key():
    key = request.headers.get('x-api-key')
    if key != API_KEY:
        return jsonify({'error': 'Unauthorized'}), 401

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    try:
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
    except Exception as e:
        return jsonify({'error': 'Invalid image', 'detail': str(e)}), 400

    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        label = labels[int(predicted.item())]
    return jsonify({'prediction': label})

if __name__ == '__main__':
    # for local testing
    app.run(host='0.0.0.0', port=5000, debug=True)
