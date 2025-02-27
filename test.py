import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Define constants
MODEL_PATH = "C:/Users/LENOVO/Desktop/Major Project/trained_model.h5"
TEST_IMAGE_PATH = "C:/Users/LENOVO/Desktop/Major Project/dataset/Test/00093.png"  # Change this to your test image
IMG_SIZE = (64, 64)

# Load trained model
print("Loading trained model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# Define class labels (Ensure these match the training dataset)
class_labels = [
    "Speed Limit 20", "Speed Limit 30", "Speed Limit 50", "Speed Limit 60", 
    "Speed Limit 70", "Speed Limit 80", "End of Speed Limit 80", "Speed Limit 100", 
    "Speed Limit 120", "No Overtaking", "No Overtaking for Trucks", 
    "Right of Way", "Priority Road", "Yield", "Stop", 
    "No Vehicles", "No Trucks", "No Entry", "General Caution", 
    "Dangerous Curve Left", "Dangerous Curve Right", "Double Curve", 
    "Bumpy Road", "Slippery Road", "Road Narrows on Right", 
    "Road Work", "Traffic Signals", "Pedestrian Crossing", "Children Crossing", 
    "Bicycles Crossing", "Beware of Ice/Snow", "Wild Animals", "End of Restrictions", 
    "Turn Right", "Turn Left", "Go Straight", "Go Right or Straight", 
    "Go Left or Straight", "Keep Right", "Keep Left", "Roundabout", 
    "End of No Overtaking", "End of No Overtaking for Trucks"
]

# Function to preprocess the image
def preprocess_image(image_path):
    """Preprocess an image for model prediction."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error: Could not read image at {image_path}")

    img = cv2.resize(img, IMG_SIZE)
    img = img_to_array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Load and preprocess the test image
print("Processing test image...")
image = preprocess_image(TEST_IMAGE_PATH)

# Make prediction
print("Predicting...")
predictions = model.predict(image)
predicted_class = np.argmax(predictions)  # Get class index
confidence = np.max(predictions)  # Get confidence score

# Get the predicted class name
sign_name = class_labels[predicted_class] if predicted_class < len(class_labels) else "Unknown"

# Print result with sign name
print(f"Predicted Sign: {sign_name}, Confidence: {confidence:.2f}")
