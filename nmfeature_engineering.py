import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Load MNIST digits (0-9)
(X_train_digits, y_train_digits), (_, _) = mnist.load_data()

# 2. Create synthetic letters (A-Z) since EMNIST requires extra installation
# Each letter is just a placeholder image (in practice, use EMNIST for real letters)
letter_images = []
for char_code in range(65, 91):  # ASCII codes for A-Z
    # Create simple letter patterns
    img = np.zeros((28, 28))
    img[5:23, 10:18] = 0.7  # Vertical bar
    if chr(char_code) in ['A', 'B', 'D', 'O', 'P', 'Q', 'R']:
        img[5:10, 12:16] = 0.9  # Horizontal bar
    letter_images.append(img)
X_train_letters = np.array(letter_images * 100)  # 100 copies of each letter
y_train_letters = np.array([i-55 for i in range(65, 91)] * 100)  # Labels 10-35

# 3. Combine datasets
X_train = np.concatenate([
    X_train_digits, 
    X_train_letters
]).reshape(-1, 28, 28, 1).astype('float32') / 255.0

y_train = np.concatenate([
    y_train_digits, 
    y_train_letters
])

# 4. Character mapping
label_to_char = {i: str(i) for i in range(10)}  # 0-9
label_to_char.update({i+10: chr(65+i) for i in range(26)})  # A-Z

# 5. Visualize samples
plt.figure(figsize=(15, 6))
for i in range(15):  # Show first 15 characters
    plt.subplot(3, 5, i+1)
    char_label = list(label_to_char.keys())[i]
    idx = np.where(y_train == char_label)[0][0]
    plt.imshow(X_train[idx].squeeze(), cmap='gray')
    plt.title(f"{char_label} ({label_to_char[char_label]})")
    plt.axis('off')
plt.suptitle('Digit and Letter Samples')
plt.tight_layout()
plt.show()

# 6. Class distribution
plt.figure(figsize=(18, 5))
char_counts = Counter(y_train)
chars = sorted(char_counts.keys())
counts = [char_counts[c] for c in chars]
plt.bar([label_to_char[c] for c in chars], counts)
plt.title('Class Distribution')
plt.xlabel('Character')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# 7. Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=8,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.08
)

# 8. Visualize augmentation
sample_idx = np.where(y_train == 10)[0][0]  # First 'A'
sample = X_train[sample_idx].reshape(1, 28, 28, 1)

plt.figure(figsize=(12, 3))
plt.suptitle(f"Augmented Samples of: A")
for i in range(5):
    batch = next(datagen.flow(sample, batch_size=1))
    plt.subplot(1, 5, i+1)
    plt.imshow(batch[0].squeeze(), cmap='gray')
    plt.axis('off')
plt.show()

# 9. Print statistics
print("\nTop 10 Character Counts:")
print(f"{'Label':<6} {'Char':<5} {'Count':<7} {'Pct':<6}")
print("-"*30)
total = sum(counts)
for c, cnt in sorted(char_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"{c:<6} {label_to_char[c]:<5} {cnt:<7} {cnt/total:.2%}")