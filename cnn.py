# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
def load_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # Normalize the pixel values to [0, 1] by dividing by 255
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Reshape to add an extra dimension (channels) required by convolutional layers
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    return train_images, train_labels, test_images, test_labels

# Build the CNN model
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Function to train the model
def train_model(model, train_images, train_labels):
    history = model.fit(train_images, train_labels, epochs=5,
                        validation_split=0.2, verbose=2)
    return history

# Function to evaluate the model
def evaluate_model(model, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")

# Function to display example predictions
def display_predictions(model, test_images, test_labels):
    predictions = model.predict(test_images[:5])

    for i in range(5):
        plt.imshow(test_images[i].reshape(28, 28), cmap=plt.cm.binary)
        plt.title(f"Predicted: {np.argmax(predictions[i])}, Actual: {test_labels[i]}")
        plt.show()

# Main function to tie everything together
def main():
    # Load the dataset
    train_images, train_labels, test_images, test_labels = load_data()

    # Build and compile the model
    model = build_model()

    # Train the model
    train_model(model, train_images, train_labels)

    # Evaluate the model on the test set
    evaluate_model(model, test_images, test_labels)

    # Display predictions for the first 5 test images
    display_predictions(model, test_images, test_labels)

if __name__ == "__main__":
    main()
