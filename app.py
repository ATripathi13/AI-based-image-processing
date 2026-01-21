import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

# Define the CNN model (Same as training)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
try:
    model.load_state_dict(torch.load('mnist_model.pth', map_location=device, weights_only=True))
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Transform
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data'}), 400

    # Decode base64 image
    img_data = data['image'].split(',')[1]
    img_bytes = base64.b64decode(img_data)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

    # 1. Binarize (Otsu's thresholding)
    # MNIST is white on black. Assume user provides black on white or capture.
    if np.mean(img) > 127:
        img = 255 - img
    
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Find bounding box and center (MNIST standard: 20x20 digit centered in 28x28)
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        digit = img[y:y+h, x:x+w]
        
        # Resize digit to fit in 20x20
        aspect = w / h
        if w > h:
            new_w = 20
            new_h = int(20 / aspect)
        else:
            new_h = 20
            new_w = int(20 * aspect)
        
        digit = cv2.resize(digit, (new_w, new_h))
        
        # Pad to 28x28
        top = (28 - new_h) // 2
        bottom = 28 - new_h - top
        left = (28 - new_w) // 2
        right = 28 - new_w - left
        img = cv2.copyMakeBorder(digit, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    else:
        # If empty
        img = cv2.resize(img, (28, 28))

    # Preprocess for model
    img_pil = Image.fromarray(img)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return jsonify({
        'prediction': int(predicted.item()),
        'confidence': float(confidence.item())
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
