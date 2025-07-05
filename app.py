import os
import numpy as np
import cv2
from flask import Flask, render_template, request, render_template_string
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from time import time
import random

emotion_group_map = {
    'happy': 'happy',
    'surprise': 'happy',
    'neutral': 'neutral',
    'sad': 'sad',
    'fear': 'sad',
    'angry': 'angry',
    'disgust': 'angry'
}

def get_audio_for_emotion(emotion_group):
    audio_dir = os.path.join('static', 'audio', emotion_group)
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(('.mp3', '.wav'))]
    if audio_files:
        return f'audio/{emotion_group}/{random.choice(audio_files)}'
    return None

app = Flask(__name__)
model = load_model("model/emotion_model.keras")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def predict_emotion(image_path):
    img_cv = cv2.imread(image_path)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Improve contrast

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30)
    )

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
    else:
        # If no face detected, use the whole image resized
        roi_gray = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)

    img_array = np.expand_dims(roi_gray, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    prediction = model.predict(img_array)
    max_index = np.argmax(prediction)
    return emotion_labels[max_index], round(float(prediction[0][max_index]) * 100, 2)


@app.route('/', methods=['GET', 'POST'])
def index():
    emotion = None
    confidence = None
    image_url = None
    audio_path = None
    timestamp = int(time() * 1000)

    if request.method == 'POST':
        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename != '':
                filename = secure_filename(file.filename)
                upload_path = os.path.join('static', 'uploads')
                os.makedirs(upload_path, exist_ok=True)
                filepath = os.path.join(upload_path, filename)
                file.save(filepath)
                emotion, confidence = predict_emotion(filepath)
                image_url = filepath
                # Map detected emotion to group and get audio
                emotion_group = emotion_group_map.get(emotion.lower(), 'neutral')
                audio_path = get_audio_for_emotion(emotion_group)

        # Handle AJAX request
        if request.headers.get('X-Requested-With') == 'fetch':
            return render_template_string("""
                {% if emotion %}
                    <h2>Detected Emotion: {{ emotion }} ({{ confidence }}%)</h2>
                    {% if image_url %}
                        <img src=\"{{ image_url }}?t={{ timestamp }}\" alt=\"Uploaded Image\" width=\"200\"><br>
                        {% if audio_path %}
                        <audio autoplay controls>
                            <source src=\"{{ url_for('static', filename=audio_path) }}\" type=\"audio/mpeg\">
                        </audio>
                        {% endif %}
                    {% endif %}
                {% endif %}
            """, emotion=emotion, confidence=confidence, image_url=image_url, timestamp=timestamp, audio_path=audio_path)

    return render_template('index.html', emotion=emotion, confidence=confidence, image_url=image_url, timestamp=timestamp, audio_path=audio_path)

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True)
    