import cv2
import numpy as np
import tensorflow as tf
import pyttsx3  # Text-to-Speech
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from collections import deque

# Constants
MODEL_PATH = "C:/Users/LENOVO/Desktop/Major Project/trained_model.h5"
IMG_SIZE = (64, 64)
CONFIDENCE_THRESHOLD = 0.75  # Ignore low-confidence predictions
FRAME_SKIP = 5  # Process every 5th frame
SMOOTHING_FRAMES = 10  # Number of frames for stable prediction
STABLE_FRAMES_REQUIRED = 10  # Speak only if same sign appears for 10 frames

# Load trained model
print("Loading trained model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # Adjust speaking speed

# Define class labels
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

# Initialize video capture
cap = cv2.VideoCapture(0)

# Queues for stabilizing predictions
prediction_queue = deque(maxlen=SMOOTHING_FRAMES)
stable_frame_count = 0
last_spoken_sign = None  # To avoid repeating speech
frame_count = 0

def preprocess_frame(frame):
    """Preprocess a frame for model prediction."""
    img = cv2.resize(frame, IMG_SIZE)
    img = img_to_array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue  # Skip frames for stability

    # Preprocess frame
    processed_frame = preprocess_frame(frame)

    # Make prediction
    predictions = model.predict(processed_frame)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    # Only consider predictions above threshold
    if confidence >= CONFIDENCE_THRESHOLD:
        prediction_queue.append(predicted_class)

    # Get the most stable prediction
    if len(prediction_queue) > 0:
        stable_prediction = max(set(prediction_queue), key=prediction_queue.count)
        sign_name = class_labels[stable_prediction]
    else:
        sign_name = "Detecting..."

    # Check if prediction is stable for multiple frames
    if sign_name == class_labels[predicted_class]:
        stable_frame_count += 1
    else:
        stable_frame_count = 0

    # Speak only if stable for `STABLE_FRAMES_REQUIRED` frames and not repeated
    if stable_frame_count >= STABLE_FRAMES_REQUIRED and last_spoken_sign != sign_name:
        print(f"Speaking: {sign_name}")
        engine.say(sign_name)
        engine.runAndWait()
        last_spoken_sign = sign_name  # Avoid repeating speech

    # Display result
    cv2.putText(frame, f"Sign: {sign_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Traffic Sign Recognition", frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
