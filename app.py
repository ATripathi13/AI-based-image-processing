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

# Refined CNN model (Same as improved train.py)
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImprovedCNN().to(device)

def load_trained_model():
    try:
        model.load_state_dict(torch.load('mnist_model.pth', map_location=device, weights_only=True))
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

load_trained_model()

# Transform
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def preprocess_image(image):
    """Advanced preprocessing: Centering and Otsu thresholding."""
    # Convert PIL to CV2 grayscale
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Otsu thresholding to binarize
    _, thresh = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Check if we need to invert (MNIST is white on black)
    # If the majority of pixels are white, invert it
    if np.mean(thresh) > 127:
        thresh = 255 - thresh

    # Find bounding box of the digit
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        digit = thresh[y:y+h, x:x+w]
        
        # Padding to make it square
        pad = max(w, h)
        square_digit = np.zeros((pad, pad), dtype=np.uint8)
        off_x = (pad - w) // 2
        off_y = (pad - h) // 2
        square_digit[off_y:off_y+h, off_x:off_x+w] = digit
        
        # Resize to 20x20 while keeping aspect ratio (MNIST style)
        # Then pad to 28x28
        digit_resized = cv2.resize(square_digit, (20, 20))
        final_img = np.zeros((28, 28), dtype=np.uint8)
        final_img[4:24, 4:24] = digit_resized
        return Image.fromarray(final_img)
    
    # Fallback to simple resize
    return image.convert('L')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data'}), 400

    # Decode base64 image
    try:
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Invalid image data: {e}'}), 400

    # Preprocess
    img_processed = preprocess_image(img)
    img_tensor = transform(img_processed).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return jsonify({
        'prediction': int(predicted.item()),
        'confidence': float(confidence.item())
    })

@app.route('/reload_model', methods=['POST'])
def reload_model_route():
    load_trained_model()
    return jsonify({'status': 'Model reloaded'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
