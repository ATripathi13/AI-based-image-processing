import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Refined CNN model (Same as train.py)
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

def load_model(model_path='mnist_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedCNN().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        print(f"Model loaded from {model_path}")
        return model, device
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def predict_image(model, device, image):
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    return predicted.item(), confidence.item()

def main():
    model, device = load_model()
    if model is None:
        return

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if len(sys.argv) < 2:
        print("No image path provided. Using a random image from MNIST test set.")
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=None, download=True)
        idx = np.random.randint(0, len(test_dataset))
        image, actual_label = test_dataset[idx] 
        print(f"Actual Label: {actual_label}")
        input_tensor = transform(image).unsqueeze(0)
        display_image = np.array(image)
    else:
        image_path = sys.argv[1]
        print(f"Loading image from {image_path}")
        image = Image.open(image_path).convert('L')
        input_tensor = transform(image).unsqueeze(0)
        display_image = np.array(image)

    predicted_class, confidence = predict_image(model, device, input_tensor)
    print(f"Prediction: {predicted_class} (Confidence: {confidence:.2f})")
    
    plt.imshow(display_image, cmap='gray')
    plt.title(f"Pred: {predicted_class}, Conf: {confidence:.2f}")
    plt.axis('off')
    plt.savefig('prediction_result.png')
    print(f"Prediction result saved to prediction_result.png")

if __name__ == "__main__":
    main()
