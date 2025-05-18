import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# =============================================
# 1. CNN for MNIST Digits (0-9)
# =============================================

print("\nMNIST Digit Classification (0-9)\n")

# Load MNIST data
(X_train_digits, y_train_digits), (X_test_digits, y_test_digits) = mnist.load_data()

# Preprocess data
X_train_digits = X_train_digits.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test_digits = X_test_digits.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train_digits = to_categorical(y_train_digits, 10)
y_test_digits = to_categorical(y_test_digits, 10)

# Create model
digit_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

digit_model.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

# Train with data augmentation
datagen = ImageDataGenerator(rotation_range=10, 
                             zoom_range=0.1,
                             width_shift_range=0.1,
                             height_shift_range=0.1)

digit_model.fit(datagen.flow(X_train_digits, y_train_digits, batch_size=32),
               epochs=15,
               validation_data=(X_test_digits, y_test_digits))

# =============================================
# 2. CNN for EMNIST Letters (A-Z)
# =============================================

print("\nEMNIST Letter Classification (A-Z)\n")

# Load EMNIST letters (you'll need to download this first)
# !pip install emnist
from emnist import extract_training_samples, extract_test_samples

X_train_letters, y_train_letters = extract_training_samples('letters')
X_test_letters, y_test_letters = extract_test_samples('letters')

# Preprocess data (EMNIST letters are 28x28 like MNIST)
X_train_letters = X_train_letters.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test_letters = X_test_letters.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# EMNIST letters has 26 classes (A-Z), but labels start from 1 (A=1, B=2,...)
y_train_letters = to_categorical(y_train_letters - 1, 26)
y_test_letters = to_categorical(y_test_letters - 1, 26)

# Create model (same architecture but output is 26 classes)
letter_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(26, activation='softmax')
])

letter_model.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

# Train with data augmentation
letter_model.fit(datagen.flow(X_train_letters, y_train_letters, batch_size=32),
               epochs=15,
               validation_data=(X_test_letters, y_test_letters))