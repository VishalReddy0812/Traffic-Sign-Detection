import cv2
import tensorflow as tf
import numpy as np
from dataset_loader import load_dataset

model = tf.keras.models.load_model("model/traffic_sign_model.h5")
DATASET_PATH = "dataset/"
_, _, label_map = load_dataset(DATASET_PATH)

def predict_traffic_sign(frame):
    img = cv2.resize(frame, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    label = np.argmax(prediction)
    class_name = list(label_map.keys())[list(label_map.values()).index(label)]
    return class_name

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    class_name = predict_traffic_sign(frame)
    cv2.putText(frame, class_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Traffic Sign Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
