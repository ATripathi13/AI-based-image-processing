# AI-Based Handwritten Digit Classifier
<table>
  <tr>
    <td style="width: 20%; border: none;">
      <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZjB0MDlhY2doazlxZWRzcjRhMXU2bXdweDFvdGd4NnE5cDRoYTlneSZlcD12MV9zdGlja2Vyc19zZWFyY2gmY3Q9cw/JOdoMjYu183JC1OqLc/giphy.gif" width="70" />
    </td>
    <td style="border: none; vertical-align: middle;">
      A high-accuracy handwritten digit classification application using PyTorch, 
      featuring a modern web frontend for real-time predictions via image upload 
      or camera capture.
    </td>
  </tr>
</table>

## 🚀 Features
- **High Accuracy**: Achieves **99.45% accuracy** on the MNIST test set.
- **Improved CNN Architecture**: Uses Batch Normalization and Dropout for robust generalization.
- **Real-time Prediction**: Web interface for uploading images or using a webcam.
- **Advanced Preprocessing**: Automatic centering and Otsu thresholding for reliable recognition of real-world inputs.
- **Git Managed**: Fully versioned repository with clean commit history.

## 🛠️ Tech Stack
- **AI/ML**: Python, PyTorch, Torchvision
- **Backend**: Flask, Flask-CORS, OpenCV, NumPy
- **Frontend**: HTML5, CSS3 (Glassmorphism), Vanilla JavaScript
- **Dataset**: MNIST (Handwritten digits 0-9)

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ATripathi13/AI-based-image-processing.git
   cd AI-based-image-processing
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### 1. Web Application (Recommended)
Run the Flask server to access the GUI:
```bash
python app.py
```
Open your browser and go to `http://127.0.0.1:5000`.

### 2. Manual Prediction
Predict from a local image file:
```bash
python predict.py path/to/image.png
```

### 3. Training
To retrain the model (requires downloading data):
```bash
python train.py
```

## 🧠 Model Architecture (ImprovedCNN)
The model is a Deep Convolutional Neural Network consisting of:
- **Feature Extraction**:
  - Conv2d (32 filters, 3x3) + BatchNorm + ReLU + MaxPool + Dropout(0.2)
  - Conv2d (64 filters, 3x3) + BatchNorm + ReLU + MaxPool + Dropout(0.3)
  - Conv2d (128 filters, 3x3) + BatchNorm + ReLU + Dropout(0.4)
- **Classifier**:
  - Fully Connected (256 neurons) + ReLU + Dropout(0.5)
  - Output Layer (10 neurons for digits 0-9)

## 📊 Verification
The system includes advanced image centering and thresholding logic in the preprocessing pipeline to ensure that captured digits match the MNIST distribution, resulting in high confidence even with noisy real-world inputs.

---
© 2026 AI Image Processing Tool.

