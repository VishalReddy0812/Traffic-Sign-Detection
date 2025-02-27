import joblib
import joblib
import pandas as pd
import numpy as np
import cv2
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array


TRAIN_CSV = "C:/Users/LENOVO/Desktop/Major Project/dataset/Train.csv"
TRAIN_DIR = "C:/Users/LENOVO/Desktop/Major Project/dataset/"  


IMG_SIZE = (64, 64)
NUM_CLASSES = 43

def load_dataset(csv_file, image_dir):
    """Loads images and labels from the dataset."""
    df = pd.read_csv(csv_file)

    
    if "ClassId" not in df.columns or "Path" not in df.columns:
        raise ValueError(f"Missing required columns in CSV. Found: {df.columns}")

    images = []
    labels = []

    for _, row in df.iterrows():
        img_filename = row["Path"].strip().replace("\\", "/")  
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
