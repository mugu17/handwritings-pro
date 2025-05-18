import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(X_train, y_train), (_, _) = mnist.load_data()

# Display first 10 digits (handwritten)
plt.figure(figsize=(10, 3))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(X_train[y_train == i][0], cmap='gray')
    plt.axis('off')
plt.suptitle("MNIST Handwritten Digits (0-9)", y=1.1)
plt.show()