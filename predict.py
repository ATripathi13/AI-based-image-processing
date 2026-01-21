import tensorflow as tf
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt

def load_model(model_path='mnist_model.keras'):
    """Loads the trained model."""
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_image(model, image):
    """Predicts the class of an image."""
    # Preprocess if needed (assuming image is already 28x28 grayscale)
    # Add batch dimension: (1, 28, 28, 1)
    img_batch = np.expand_dims(image, axis=0) 
    if len(img_batch.shape) == 3:
         img_batch = np.expand_dims(img_batch, axis=-1)

    prediction = model.predict(img_batch)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return predicted_class, confidence

def main():
    model = load_model()
    if model is None:
        return

    # Option 1: Load a random image from test set if no file provided
    if len(sys.argv) < 2:
        print("No image path provided. Using a random image from MNIST test set.")
        (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        # Pick random index
        idx = np.random.randint(0, len(x_test))
        image = x_test[idx]
        actual_label = y_test[idx]
        
        # Normalize
        image_norm = image / 255.0
        
        print(f"Actual Label: {actual_label}")
        
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
        # simplistic check: if mean is > 127, assume white background and invert
        if np.mean(image) > 127:
             image = 255 - image
             
        # Normalize
        image_norm = image / 255.0

    # Predict
    predicted_class, confidence = predict_image(model, image_norm)
    
    print(f"Prediction: {predicted_class} (Confidence: {confidence:.2f})")
    
    # Display result
    plt.imshow(image, cmap='gray')
    plt.title(f"Pred: {predicted_class}, Conf: {confidence:.2f}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
