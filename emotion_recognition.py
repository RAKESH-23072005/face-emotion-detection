from click import Argument
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2

# Argument parsing
parser = argparse.ArgumentParser(description="Facial Emotion Recognition")
parser.add_a
Argument('--image', type=str, required=True, help='Path to the image file')
args = parser.parse_args()

model = tf.keras.models.load_model('model/emotion_model.keras')

img_path = args.image

img_cv = cv2.imread(img_path)
gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

if gray.shape == (48, 48):
    face_img_resized = gray
else:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_img = gray[y:y+h, x:x+w]
        face_img_resized = cv2.resize(face_img, (48, 48))
    else:
        # No face detected, use the whole image resized
        face_img_resized = cv2.resize(gray, (48, 48))

# Prepare for prediction
img_array = np.expand_dims(face_img_resized, axis=-1)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
print(f"Emotion predicted: {class_names[predicted_class]}")

plt.imshow(face_img_resized, cmap="gray")
plt.title(f"Emotion: {class_names[predicted_class]}")
plt.axis('off')
plt.show()