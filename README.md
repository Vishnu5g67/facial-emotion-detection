# Facial Emotion AI - Real-time Expression Recognition

A real-time deep learning computer vision system that classifies facial expressions into 7 distinct human emotions. Features both a **Desktop OpenCV GUI** and a **Web Dashboard** deployable to cloud platforms like **Render**.

---

## 🎯 Supported Emotions
- 🟢 **Happy**
- 🔴 **Angry**
- 🟡 **Surprise**
- ⚪ **Neutral**
- 🔵 **Sad**
- 🟣 **Fear**
- 🟠 **Disgust**

---

## 📁 Project Structure

```text
facial-emotion-detection/
├── web_app.py                         # Flask web server (for local & cloud deployment)
├── templates/
│   └── index.html                     # Responsive web dashboard with live camera & upload
├── app.py                             # Desktop webcam detector (OpenCV GUI)
├── predict_image.py                   # CLI tool to predict emotions on static photos
├── Procfile                           # Render / Heroku process declaration
├── render.yaml                        # 1-click Render deployment blueprint
├── requirements.txt                   # Production dependencies
├── run.bat                            # 1-click Windows desktop launcher
├── haarcascade_frontalface_default.xml # Face detection cascade
├── model/
│   ├── emotion_model.json             # CNN model architecture
│   └── emotion_model.h5               # Trained weights (~4.2M parameters)
└── sample_images/                     # Sample images for testing
```

---

## 🌐 Deploy to Render (Free Cloud Hosting)

Deploy your web application to Render in minutes:

1. **Push this repository to GitHub** (see Git instructions below).
2. Go to [Render.com](https://render.com) and log in.
3. Click **"New +"** → **"Web Service"**.
4. Connect your GitHub repository (`facial-emotion-detection`).
5. Configure the service settings:
   * **Name:** `facial-emotion-ai` (or your choice)
   * **Language / Environment:** `Python`
   * **Region:** Any (e.g. Frankfurt, Oregon, Singapore)
   * **Branch:** `main`
   * **Build Command:**
     ```bash
     pip install -r requirements.txt
     ```
   * **Start Command:**
     ```bash
     gunicorn web_app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120
     ```
   * **Environment Variables:**
     * `PYTHON_VERSION` = `3.11.9`
6. Click **"Create Web Service"**.
7. Render will build and deploy your application. Once live, open your public `.onrender.com` URL in your browser!

---

## 💻 Running Locally

### 1. Web Application (Browser Interface)
```bash
python web_app.py
```
Open [http://localhost:5000](http://localhost:5000) in your web browser.

### 2. Desktop Application (Direct Webcam Window)
* **Windows (Double-click):** Run `run.bat`
* **Terminal:**
  ```bash
  python app.py
  ```
  *(Press `q` to exit).*

### 3. Static Image Prediction (CLI)
```bash
python predict_image.py
# Or for a specific image:
python predict_image.py path/to/image.jpg
```

---

## 📦 Dependencies
```bash
pip install -r requirements.txt
```
