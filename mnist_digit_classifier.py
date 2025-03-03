# Importing necessary libraries
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset
mnist = keras.datasets.mnist
(train_features, train_labels), (test_features, test_labels) = mnist.load_data()

# Checking the shape of train and test data
print("Train Features Shape:", train_features.shape)  # Expected: (60000, 28, 28)
print("Test Features Shape:", test_features.shape)    # Expected: (10000, 28, 28)

# Display a sample image from the dataset
id = 1000
img = train_features[id]
print("Label:", train_labels[id])  # Print the actual label of the image
plt.gray()  # Display the image in grayscale
plt.imshow(img)  # Show the selected image

# Normalize the image pixel values to be between 0 and 1
train_features = train_features / 255.0
test_features = test_features / 255.0

# Define the Neural Network model
model = keras.Sequential()

# Flatten the 28x28 images into a 1D array (784 values)
model.add(keras.layers.Flatten(input_shape=(28, 28)))

# Add two dense layers with 128 neurons and ReLU activation
model.add(keras.layers.Dense(units=128, activation='relu'))
model.add(keras.layers.Dense(units=128, activation='relu'))

# Output layer with 10 neurons (one for each digit) and softmax activation for classification
model.add(keras.layers.Dense(units=10, activation='softmax'))

# Compile the model with Adam optimizer and sparse categorical crossentropy loss function
model.compile(optimizer=tf.optimizers.Adam(),
              loss=tf.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

# Build the model explicitly for better summary visualization
model.build(input_shape=(None, 28, 28))
model.summary()  # Display the model architecture

# Train the model using the training dataset
hist = model.fit(train_features, train_labels, epochs=10, batch_size=256,
                 validation_data=(test_features, test_labels))

# Evaluate the trained model on the test dataset
test_loss, test_acc = model.evaluate(test_features, test_labels)
print("Test Accuracy:", test_acc)

# Select a test image to make a prediction
img = test_features[100]
print("Test Image Shape:", img.shape, "| True Label:", test_labels[100])

# Reshape the test image to match the input format for the model
test_data = np.reshape(img, (-1, 784))  # Convert 28x28 to a single row
print("Reshaped Test Data Shape:", test_data.shape)  # Expected: (1, 784)

# Make a prediction using the trained model
predictions = model.predict(test_data)
print("Predicted Class Probabilities:", predictions)

# Plot the training accuracy and validation accuracy
acc = hist.history["accuracy"]
val_accuracy = hist.history["val_accuracy"]

plt.figure(figsize=(8, 5))
plt.plot(acc, color="red", label="Training Accuracy")
plt.plot(val_accuracy, color="blue", label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()
