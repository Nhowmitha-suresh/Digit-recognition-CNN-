# ==============================================
# DIGIT RECOGNITION (CNN MODEL)
# Author: Nhowmitha Suresh
# ==============================================

# Importing required libraries
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# ==============================================
# STEP 1: Load Dataset
# ==============================================
print("📥 Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f"✅ Training data shape: {X_train.shape}")
print(f"✅ Test data shape: {X_test.shape}")

# ==============================================
# STEP 2: Preprocess Data
# ==============================================
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32') / 255
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32') / 255

# One-hot encode labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# ==============================================
# STEP 3: Build CNN Model
# ==============================================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

# ==============================================
# STEP 4: Compile Model
# ==============================================
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ==============================================
# STEP 5: Train Model
# ==============================================
print("\n🚀 Training the CNN model...")
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=1
)

# ==============================================
# STEP 6: Evaluate Model
# ==============================================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")

# ==============================================
# STEP 7: Predict on Test Data
# ==============================================
predictions = model.predict(X_test)
print("\n🔍 Sample Predictions:")
for i in range(5):
    predicted_label = np.argmax(predictions[i])
    actual_label = np.argmax(y_test[i])
    print(f"Sample {i+1} - Predicted: {predicted_label}, Actual: {actual_label}")

# ==============================================
# STEP 8: Plot 10 Sample Predictions
# ==============================================
def plot_images(images, labels, predictions=None):
    plt.figure(figsize=(10,5))
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(images[i].reshape(28,28), cmap='gray')
        title = f"Label: {np.argmax(labels[i])}"
        if predictions is not None:
            title += f"\nPred: {np.argmax(predictions[i])}"
        plt.title(title, fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

plot_images(X_test, y_test, predictions)

# ==============================================
# STEP 9: Test with a Custom Image
# ==============================================
custom_image_path = r"C:\Users\HP\Downloads\pred.png"  # ✅ Update path if needed

if os.path.exists(custom_image_path):
    print("\n🧠 Predicting on your custom image...")
    img = cv2.imread(custom_image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = 255 - img  # Invert if white background
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.title(f"Predicted Digit: {predicted_class}", fontsize=14)
    plt.axis('off')
    plt.show()
    print(f"✅ Predicted Digit for custom image: {predicted_class}")
else:
    print("\n⚠️ Custom image not found. Please check the path and try again.")

# ==============================================
# STEP 10: Save Model
# ==============================================
os.makedirs("model", exist_ok=True)
model.save("model/digit_model.keras")
print("\n💾 Model saved successfully at: model/digit_model.keras")

# Optionally also save as .h5 for compatibility
model.save("model/digit_model.h5", save_format="h5")
print("💾 Backup model saved as: model/digit_model.h5")
