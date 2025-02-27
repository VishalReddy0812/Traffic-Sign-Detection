import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array

print("Loading dataset...")
TRAIN_CSV = "C:/Users/LENOVO/Desktop/Major Project/dataset/Train.csv"
TRAIN_DIR = "C:/Users/LENOVO/Desktop/Major Project/dataset/"  

# Image settings
IMG_SIZE = (64, 64)
NUM_CLASSES = 43

def load_dataset(csv_file, image_dir):
    """Loads images and labels from the dataset."""
    df = pd.read_csv(csv_file)

    if "ClassId" not in df.columns or "Path" not in df.columns:
        raise ValueError(f"Missing required columns in CSV. Found: {df.columns}")

    images, labels = [], []

    for _, row in df.iterrows():
        img_filename = row["Path"].strip()
        img_path = os.path.join(image_dir, img_filename)

        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found, skipping.")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to read {img_path}, skipping.")
            continue

        img = cv2.resize(img, IMG_SIZE)
        img = img_to_array(img) / 255.0  # Normalize

        images.append(img)
        labels.append(row["ClassId"])

    images = np.array(images, dtype="float32")
    labels = np.array(labels, dtype="int")

    print(f"Loaded {len(images)} images and {len(labels)} labels.")

    if len(labels) == 0:
        raise ValueError("No valid labels found in dataset!")

    labels = to_categorical(labels, NUM_CLASSES)
    return images, labels

# Load dataset
X_train, y_train = load_dataset(TRAIN_CSV, TRAIN_DIR)
print(f"Final Dataset Shape: X_train={X_train.shape}, y_train={y_train.shape}")

# Define CNN model
print("Building model...")
model = Sequential([
    Input(shape=(64, 64, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
print("Starting training...")
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# Save model
model.save("C:/Users/LENOVO/Desktop/Major Project/trained_model.h5")
print("Model saved successfully!")
print("Training completed!")
