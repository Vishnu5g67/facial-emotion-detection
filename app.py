import os
import sys
import cv2
import numpy as np

# Set TensorFlow log level to suppress non-critical warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import model_from_json

# Resolve base directory path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_JSON_PATH = os.path.join(BASE_DIR, 'model', 'emotion_model.json')
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, 'model', 'emotion_model.h5')
HAAR_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')

# Emotion labels mapping
LABELS = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'
}

# Emotion color themes (BGR format for OpenCV)
EMOTION_COLORS = {
    'Happy': (0, 230, 0),        # Green
    'Sad': (255, 128, 0),        # Blue/Cyan
    'Angry': (0, 0, 255),        # Red
    'Surprise': (0, 255, 255),    # Yellow
    'Neutral': (200, 200, 200),  # Light Gray
    'Fear': (255, 0, 128),       # Purple
    'Disgust': (0, 140, 255)     # Orange
}


def load_emotion_model():
    """Loads the pre-trained emotion detection CNN model."""
    if not os.path.exists(MODEL_JSON_PATH) or not os.path.exists(MODEL_WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Model files not found in '{os.path.join(BASE_DIR, 'model')}'. "
            "Please ensure emotion_model.json and emotion_model.h5 are present."
        )

    print("Loading emotion detection model...")
    with open(MODEL_JSON_PATH, 'r') as f:
        model_json = f.read()

    model = model_from_json(model_json)
    model.load_weights(MODEL_WEIGHTS_PATH)
    print("Model loaded successfully!")
    return model


def load_face_detector():
    """Loads the Haar Cascade face detector."""
    cascade_path = HAAR_PATH
    if not os.path.exists(cascade_path):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load Haar Cascade face detector from '{cascade_path}'.")
    return face_cascade


def preprocess_face(face_gray):
    """Resizes and normalizes the cropped face for CNN input (48x48x1)."""
    resized = cv2.resize(face_gray, (48, 48))
    feature = np.array(resized, dtype='float32') / 255.0
    return feature.reshape(1, 48, 48, 1)


def main():
    try:
        model = load_emotion_model()
        face_cascade = load_face_detector()
    except Exception as e:
        print(f"Initialization Error: {e}", file=sys.stderr)
        return

    # Initialize webcam (index 0)
    print("Opening webcam... Press 'q' to quit.")
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not access the webcam. Check camera permissions or device connection.", file=sys.stderr)
        return

    window_name = "Facial Emotion Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab webcam frame.", file=sys.stderr)
            break

        # Flip horizontally for natural mirror feel
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            input_tensor = preprocess_face(face_roi)

            # Predict emotion
            predictions = model.predict(input_tensor, verbose=0)[0]
            label_idx = np.argmax(predictions)
            label = LABELS.get(label_idx, 'Unknown')
            confidence = predictions[label_idx] * 100.0

            color = EMOTION_COLORS.get(label, (0, 255, 0))
            display_text = f"{label} ({confidence:.1f}%)"

            # Draw bounding box around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Draw background tag for legible text
            (text_w, text_h), baseline = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            tag_y = max(y - 10, text_h + 10)
            cv2.rectangle(frame, (x, tag_y - text_h - 6), (x + text_w + 6, tag_y + 4), color, cv2.FILLED)
            cv2.putText(frame, display_text, (x + 3, tag_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        # Draw footer instructions
        cv2.putText(
            frame,
            "Press 'q' to Quit",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.imshow(window_name, frame)

        # Press 'q' or 'ESC' to exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    camera.release()
    cv2.destroyAllWindows()
    print("Application closed.")


if __name__ == '__main__':
    main()
