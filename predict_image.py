import os
import sys
import cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import model_from_json

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


def load_model():
    with open(MODEL_JSON_PATH, 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(MODEL_WEIGHTS_PATH)
    return model


def predict_single_image(model, image_path):
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not decode image: {image_path}")
        return

    # Resize to 48x48
    img_resized = cv2.resize(img, (48, 48))
    tensor = (img_resized.reshape(1, 48, 48, 1) / 255.0).astype('float32')

    preds = model.predict(tensor, verbose=0)[0]
    best_idx = np.argmax(preds)
    best_label = LABELS[best_idx]
    confidence = preds[best_idx] * 100.0

    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"--> Predicted Emotion: {best_label.upper()} ({confidence:.1f}% confidence)")
    print("Class Probabilities:")
    for idx, name in LABELS.items():
        bar = "#" * int(preds[idx] * 30)
        print(f"  {name:10}: {preds[idx] * 100.0:5.1f}%  {bar}")


def main():
    model = load_model()
    sample_dir = os.path.join(BASE_DIR, 'sample_images')

    if len(sys.argv) > 1:
        target_path = sys.argv[1]
        predict_single_image(model, target_path)
    else:
        print("No image provided. Running predictions on all sample images in 'sample_images/'...")
        if os.path.exists(sample_dir):
            for file in os.listdir(sample_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    predict_single_image(model, os.path.join(sample_dir, file))
        else:
            print("Usage: python predict_image.py <path_to_image>")


if __name__ == '__main__':
    main()
