import os
import sys
import time
import base64
import cv2
import numpy as np

# Configure single-thread execution before TensorFlow import to prevent Gunicorn thread deadlocks on Linux
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

import tensorflow as tf
try:
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
except Exception:
    pass

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
_cascade_source = 'none'


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
    global _face_cascade, _cascade_source
    if _face_cascade is not None:
        return _face_cascade

    # Safely resolve CascadeClassifier across various OpenCV builds
    CascadeClassifierClass = None
    if hasattr(cv2, 'CascadeClassifier'):
        CascadeClassifierClass = cv2.CascadeClassifier
    elif hasattr(getattr(cv2, 'objdetect', None), 'CascadeClassifier'):
        CascadeClassifierClass = cv2.objdetect.CascadeClassifier

    if CascadeClassifierClass is None:
        print("Warning: cv2 has no CascadeClassifier attribute in this build", file=sys.stderr)
        _cascade_source = 'unsupported_cv2_build'
        return None

    # Safe built-in cascade path check
    data_dir = getattr(cv2, 'data', None)
    haarcascades_dir = getattr(data_dir, 'haarcascades', None) if data_dir else None
    builtin_xml = (haarcascades_dir + 'haarcascade_frontalface_default.xml') if haarcascades_dir else None

    candidates = [
        builtin_xml,
        HAAR_PATH,
        os.path.abspath('haarcascade_frontalface_default.xml'),
        '/tmp/haarcascade_frontalface_default.xml'
    ]

    for p in candidates:
        if p and os.path.exists(p):
            try:
                cas = CascadeClassifierClass(p)
                if not cas.empty():
                    _face_cascade = cas
                    _cascade_source = p
                    print(f"Haar cascade loaded successfully from: {p}")
                    return _face_cascade
            except Exception as e:
                print(f"Failed to load candidate {p}: {e}", file=sys.stderr)

    # Fallback: Download official cascade to /tmp if all local paths failed
    try:
        import urllib.request
        tmp_target = '/tmp/haarcascade_frontalface_default.xml'
        if not os.path.exists(tmp_target):
            url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
            print(f"Attempting to download cascade from {url}...")
            urllib.request.urlretrieve(url, tmp_target)

        if os.path.exists(tmp_target):
            cas = CascadeClassifierClass(tmp_target)
            if not cas.empty():
                _face_cascade = cas
                _cascade_source = tmp_target
                print("Haar cascade downloaded and loaded successfully.")
                return _face_cascade
    except Exception as ex:
        print(f"Fallback cascade download failed: {ex}", file=sys.stderr)

    return _face_cascade


def detect_faces(gray):
    """Fast, multi-stage face detection with sensitivity and center-crop fallbacks."""
    h, w = gray.shape[:2]
    # If the image is already a small cropped face avatar
    if w <= 140 and h <= 140:
        return [(0, 0, w, h)]

    cascade = get_face_cascade()
    if cascade is None or cascade.empty():
        # Fallback to center portrait region if cascade is unavailable
        cw = int(w * 0.6)
        ch = int(h * 0.6)
        cx = int((w - cw) / 2)
        cy = int(h * 0.15)
        return [(cx, cy, cw, ch)]

    # Downscale for sub-15ms Haar cascade execution
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

    # Pass 2: Higher sensitivity fallback if no face detected in Pass 1
    if len(faces) == 0:
        faces = cascade.detectMultiScale(
            small_gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(20, 20)
        )

    # Pass 3: Ultra sensitive fallback for tricky angles or glasses
    if len(faces) == 0:
        faces = cascade.detectMultiScale(
            small_gray,
            scaleFactor=1.03,
            minNeighbors=2,
            minSize=(15, 15)
        )

    # If still 0 faces detected on an uploaded portrait, treat center area as face
    if len(faces) == 0 and (w < 800 and h < 800):
        cw = int(w * 0.5)
        ch = int(h * 0.5)
        cx = int((w - cw) / 2)
        cy = int(h * 0.15)
        return [(cx, cy, cw, ch)]

    # Scale coordinates back up to original image resolution
    results = []
    for (sx, sy, sw, sh) in faces:
        x = int(round(sx / scale))
        y = int(round(sy / scale))
        bw = int(round(sw / scale))
        bh = int(round(sh / scale))
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
    try:
        cas = get_face_cascade()
        return jsonify({
            'status': 'healthy',
            'service': 'facial-emotion-detection',
            'model_loaded': _model is not None,
            'cascade_loaded': cas is not None and not cas.empty(),
            'cascade_source': _cascade_source,
            'cv2_version': getattr(cv2, '__version__', 'unknown')
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


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
        import traceback
        traceback.print_exc()
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
