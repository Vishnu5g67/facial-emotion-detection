import base64
import os
import sys
import time
import cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import model_from_json
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)

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

_model = None
_face_cascade = None


def get_model():
    global _model
    if _model is None:
        print("Loading emotion detection CNN model...")
        with open(MODEL_JSON_PATH, 'r') as f:
            _model = model_from_json(f.read())
        _model.load_weights(MODEL_WEIGHTS_PATH)
        # Warmup forward pass
        dummy = np.zeros((1, 48, 48, 1), dtype='float32')
        _ = _model(dummy, training=False)
        print("Emotion model loaded & warmed up successfully.")
    return _model


def get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        cascade_path = HAAR_PATH
        if not os.path.exists(cascade_path):
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        _face_cascade = cv2.CascadeClassifier(cascade_path)
        if _face_cascade.empty():
            print("Warning: Haar cascade could not be loaded from path!", file=sys.stderr)
        else:
            print("Haar cascade loaded successfully.")
    return _face_cascade


def detect_faces(gray):
    """
    Fast and multi-pass face detection:
    - Scales down large images for sub-10ms Haar cascade execution.
    - Handles pre-cropped face images gracefully.
    - Uses sensitive fallback thresholds if initial pass finds no face.
    """
    h, w = gray.shape[:2]
    # If the image is already a small cropped face (e.g. sample images or tight avatar)
    if w <= 140 and h <= 140:
        return [(0, 0, w, h)]

    cascade = get_face_cascade()
    if cascade.empty():
        return [(0, 0, w, h)]

    # Downscale for fast cascade computation
    scale = 1.0
    max_dim = 360
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        small_gray = cv2.resize(gray, (int(w * scale), int(h * scale)))
    else:
        small_gray = gray

    # Pass 1: Standard detection
    faces = cascade.detectMultiScale(
        small_gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(25, 25)
    )

    # Pass 2: High sensitivity fallback if no faces were caught
    if len(faces) == 0:
        faces = cascade.detectMultiScale(
            small_gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(20, 20)
        )

    # Scale coordinates back up to original image resolution
    results = []
    for (sx, sy, sw, sh) in faces:
        x = int(round(sx / scale))
        y = int(round(sy / scale))
        bw = int(round(sw / scale))
        bh = int(round(sh / scale))
        # Ensure within image boundaries
        x = max(0, x)
        y = max(0, y)
        bw = min(bw, w - x)
        bh = min(bh, h - y)
        results.append((x, y, bw, bh))

    return results


def process_image(img_bgr):
    """Detects faces in BGR image and predicts emotion for each face."""
    model = get_model()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    face_boxes = detect_faces(gray)

    results = []
    for (x, y, w, h) in face_boxes:
        if w < 10 or h < 10:
            continue
        face_roi = gray[y:y + h, x:x + w]
        resized = cv2.resize(face_roi, (48, 48))
        tensor = (resized.reshape(1, 48, 48, 1) / 255.0).astype('float32')

        # Fast forward pass
        preds = model(tensor, training=False).numpy()[0]
        best_idx = int(np.argmax(preds))
        best_label = LABELS.get(best_idx, 'Neutral')
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


@app.route('/sample_images/<filename>')
def serve_sample(filename):
    sample_dir = os.path.join(BASE_DIR, 'sample_images')
    return send_from_directory(sample_dir, filename)


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'facial-emotion-detection',
        'model_loaded': _model is not None,
        'cascade_loaded': _face_cascade is not None and not _face_cascade.empty()
    })


@app.route('/api/detect', methods=['POST'])
def detect():
    start_time = time.time()
    try:
        data = request.get_json(silent=True)
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]

        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        h, w = img.shape[:2]
        faces = process_image(img)
        elapsed_ms = round((time.time() - start_time) * 1000, 1)

        return jsonify({
            'success': True,
            'width': w,
            'height': h,
            'face_count': len(faces),
            'faces': faces,
            'latency_ms': elapsed_ms
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Warmup at module import time
try:
    get_model()
    get_face_cascade()
except Exception as e:
    print(f"Warmup notice: {e}", file=sys.stderr)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Server starting on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
