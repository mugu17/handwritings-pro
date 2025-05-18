import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Load only MNIST digits (0-9) since EMNIST requires separate installation
(X_train, y_train), (_, _) = mnist.load_data()

# 2. Preprocess data
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# 3. Character mapping (digits only for this example)
label_to_char = {i: str(i) for i in range(10)}

# 4. Visualize samples
plt.figure(figsize=(10, 5))
for i in range(10):  # Show first 10 digits
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train[i].squeeze(), cmap='gray')
    plt.title(f"Label: {label_to_char[y_train[i]]}")
    plt.axis('off')
plt.suptitle('MNIST Digit Samples')
plt.tight_layout()
plt.show()

# 5. Class distribution
plt.figure(figsize=(10, 5))
digit_counts = Counter(y_train)
digits = sorted(digit_counts.keys())
counts = [digit_counts[d] for d in digits]
plt.bar(digits, counts)
plt.title('Digit Class Distribution')
plt.xlabel('Digit')
plt.ylabel('Count')
plt.xticks(digits)
plt.show()

# 6. Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=8,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.08
)

# 7. Visualize augmentation for digit '5'
sample_idx = np.where(y_train == 5)[0][0]  # First '5'
sample = X_train[sample_idx].reshape(1, 28, 28, 1)

plt.figure(figsize=(12, 3))
plt.suptitle('Augmented Samples of Digit 5')
for i in range(5):
    batch = next(datagen.flow(sample, batch_size=1))  # Fixed typo from previous version
    plt.subplot(1, 5, i+1)
    plt.imshow(batch[0].squeeze(), cmap='gray')
    plt.axis('off')
plt.show()

# 8. Print statistics
print("\nDigit Count Statistics:")
print(f"{'Digit':<6} {'Count':<7} {'Percentage':<10}")
print("-"*25)
total = sum(counts)
for d, cnt in zip(digits, counts):
    print(f"{d:<6} {cnt:<7} {cnt/total:.2%}")