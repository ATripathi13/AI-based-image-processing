import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

def load_data():
    """Loads and preprocesses the MNIST dataset."""
    print("Loading MNIST dataset...")
    # Load dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Add a channel dimension (images are grayscale, so 1 channel)
    # Shape becomes (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    return (x_train, y_train), (x_test, y_test)

def create_model():
    """Creates a simple CNN model."""
    print("Creating model...")
    model = models.Sequential([
        # Convolutional base
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Dense top
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax') # 10 output classes for digits 0-9
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    # 1. Load Data
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # 2. Create Model
    model = create_model()
    model.summary()
    
    # 3. Train Model
    print("Starting training...")
    history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    
    # 4. Evaluate
    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTest accuracy: {test_acc}")
    
    # 5. Save Model
    model_path = 'mnist_model.keras'
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
