import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

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

def load_model(model_path='mnist_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        print(f"Model loaded from {model_path}")
        return model, device
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def predict_image(model, device, image):
    """Predicts the class of an image tensor."""
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        # Apply softmax to get probabilities (optional, since we just need max for class)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    return predicted.item(), confidence.item()

def main():
    model, device = load_model()
    if model is None:
        return

    # Transform for the input image (same as training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Option 1: Load a random image from test set if no file provided
    if len(sys.argv) < 2:
        print("No image path provided. Using a random image from MNIST test set.")
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=None, download=True)
        
        # Pick random index
        idx = np.random.randint(0, len(test_dataset))
        image, actual_label = test_dataset[idx] # image is PIL Image
        
        print(f"Actual Label: {actual_label}")
        
        # Prepare for model
        input_tensor = transform(image).unsqueeze(0) # Add batch dimension -> (1, 1, 28, 28)
        
        # For display
        display_image = np.array(image)

    # Option 2: Load specific file
    else:
        image_path = sys.argv[1]
        print(f"Loading image from {image_path}")
        
        # Read as grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print("Could not read image.")
            return

        # Resize to 28x28 if not already
        if image.shape != (28, 28):
            image = cv2.resize(image, (28, 28))
            
        # Invert colors if needed (MNIST is white on black, typical images are black on white)
        if np.mean(image) > 127:
             image = 255 - image
        
        # Prepare for model
        # Convert to PIL Image or numpy array [0, 255] then ToTensor
        # ToTensor scales [0, 255] to [0.0, 1.0]
        image_norm = image / 255.0
        input_tensor = torch.tensor(image_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # (1, 1, 28, 28)
        # Normalize with same mean/std as training: (input - 0.5) / 0.5
        input_tensor = (input_tensor - 0.5) / 0.5
        
        display_image = image

    # Predict
    predicted_class, confidence = predict_image(model, device, input_tensor)
    
    print(f"Prediction: {predicted_class} (Confidence: {confidence:.2f})")
    
    # Display result
    plt.imshow(display_image, cmap='gray')
    plt.title(f"Pred: {predicted_class}, Conf: {confidence:.2f}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
