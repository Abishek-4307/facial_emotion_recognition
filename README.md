# 🎭 Facial Emotion Recognition Studio

A modern, high-performance web interface built with **Streamlit**, **TensorFlow**, and **OpenCV** for real-time facial emotion recognition and affective computing analysis.

---

## 🌟 Key Features

- **📸 Live Camera Studio**:
  - **Real-Time OpenCV Webcam Stream**: Continuous video feed with live detection brackets, confidence metrics, and dynamic FPS counter (latest detections always displayed first).
  - **Instant Camera Snapshot**: Browser camera photo capture with full emotional probability breakdown.
- **📁 Multi-Face Image Inspection**:
  - Drag-and-drop support for JPG, PNG, WEBP.
  - Detects and numbers multiple faces in group portraits.
  - Individual face avatar crops and confidence meters.
  - Downloadable annotated results with one click.
- **📈 Emotion Analytics & Timeline**:
  - **Emotion Affect Radar**: Clean polar radar chart visualizing affective dimensions across all 7 emotions.
  - **Affective Valence Split**: Positive vs Neutral vs Negative breakdown.
  - **Session Detection Log**: Chronological log (latest first) with export to CSV.
- **🎨 Swiss Minimal Design (International Typographic Style)**:
  - Deep matte obsidian palette (`#0A0B0E`), Swiss Red (`#EF4444`) functional accents, clean tabular metrics, and disciplined typography (`Inter` & `JetBrains Mono`).

---

## 🧠 Model Specifications

- **Architecture**: Deep Convolutional Neural Network (CNN)
- **Input Dimension**: `(1, 48, 48, 1)` Grayscale normalized tensor
- **Target Classes**: 7 Emotions (`Angry`, `Disgust`, `Fear`, `Happy`, `Neutral`, `Sad`, `Surprise`)
- **Face Detection**: OpenCV Haar Feature-based Cascade Classifier

---

## 🚀 Quick Start

### 1. Launch via Terminal
```bash
streamlit run app.py
```

### 2. Launch via Batch File (Windows)
Double-click `run_app.bat` in the project root directory.

The web interface will automatically open in your default browser at `http://localhost:8501`.

---

## 📂 Project Structure

```text
FacialEmotionRecognition/
├── app.py                     # Main Streamlit web application
├── emotion_utils.py           # Neural inference, face detection, & HUD drawing
├── webcam_emotion.py          # Original OpenCV CLI webcam script
├── run_app.bat                # Windows one-click launcher
├── models/
│   └── emotion_cnn.keras      # Trained TensorFlow emotion model
├── samples/                   # Pre-generated sample portraits
│   ├── happy.jpg
│   ├── surprise.jpg
│   ├── neutral.jpg
│   └── angry.jpg
└── .streamlit/
    └── config.toml            # Custom dark theme configuration
```
