import base64
import os
import sys
import cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import model_from_json
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_JSON_PATH = os.path.join(BASE_DIR, 'model', 'emotion_model.json')
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, 'model', 'emotion_model.h5')
HAAR_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')

LABELS = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'
}

EMOTION_COLORS = {
    'Happy': '#10b981',    # Emerald Green
    'Sad': '#3b82f6',      # Blue
    'Angry': '#ef4444',    # Red
    'Surprise': '#f59e0b', # Amber / Yellow
    'Neutral': '#9ca3af',  # Gray
    'Fear': '#a855f7',     # Purple
    'Disgust': '#f97316'   # Orange
}

# Globals for lazy/cached model loading
_model = None
_face_cascade = None


def get_model():
    global _model
    if _model is None:
        print("Loading emotion detection model...")
        with open(MODEL_JSON_PATH, 'r') as f:
            _model = model_from_json(f.read())
        _model.load_weights(MODEL_WEIGHTS_PATH)
        print("Model loaded successfully.")
    return _model


def get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        cascade_path = HAAR_PATH
        if not os.path.exists(cascade_path):
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


def process_image(img_bgr):
    """Detects faces in BGR image and predicts emotion for each face."""
    model = get_model()
    face_cascade = get_face_cascade()

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30)
    )

    results = []
    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        resized = cv2.resize(face_roi, (48, 48))
        tensor = (resized.reshape(1, 48, 48, 1) / 255.0).astype('float32')

        preds = model.predict(tensor, verbose=0)[0]
        best_idx = int(np.argmax(preds))
        best_label = LABELS.get(best_idx, 'Unknown')
        confidence = float(preds[best_idx] * 100.0)

        scores = {LABELS[i]: round(float(preds[i] * 100.0), 1) for i in range(len(LABELS))}

        results.append({
            'box': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
            'emotion': best_label,
            'confidence': round(confidence, 1),
            'color': EMOTION_COLORS.get(best_label, '#10b981'),
            'scores': scores
        })

    return results


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'service': 'facial-emotion-detection'})


@app.route('/api/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json(silent=True)
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # Extract base64 payload
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]

        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        h, w, _ = img.shape
        faces = process_image(img)

        return jsonify({
            'success': True,
            'width': w,
            'height': h,
            'face_count': len(faces),
            'faces': faces
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Eagerly load model on startup
    get_model()
    get_face_cascade()
    port = int(os.environ.get('PORT', 5000))
    print(f"Server starting on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
