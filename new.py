import cv2
img_path = "C:/Users/LENOVO/Desktop/Major Project/dataset/Train/0/00000_00000_00002.png"
img = cv2.imread(img_path)
if img is None:
    print("Image not found or corrupted!")
else:
    print("Image loaded successfully!")
