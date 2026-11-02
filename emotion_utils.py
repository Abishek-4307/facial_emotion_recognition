import cv2
import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple, Any

# Emotion configuration mapping
CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

EMOTION_META = {
    "angry": {
        "label": "Angry",
        "emoji": "😡",
        "color": "#FF4D4D",
        "bgr": (77, 77, 255),
        "valence": "Negative",
        "description": "High arousal negative emotion, tension in facial musculature."
    },
    "disgust": {
        "label": "Disgust",
        "emoji": "🤢",
        "color": "#00E676",
        "bgr": (118, 230, 0),
        "valence": "Negative",
        "description": "Aversion response, wrinkled nose and raised upper lip."
    },
    "fear": {
        "label": "Fear",
        "emoji": "😨",
        "color": "#B388FF",
        "bgr": (255, 136, 179),
        "valence": "Negative",
        "description": "Alertness to potential threat, widened eyes and raised brows."
    },
    "happy": {
        "label": "Happy",
        "emoji": "😊",
        "color": "#FFD600",
        "bgr": (0, 214, 255),
        "valence": "Positive",
        "description": "Positive affect, zygomaticus major activation and smiling."
    },
    "neutral": {
        "label": "Neutral",
        "emoji": "😐",
        "color": "#90CAF9",
        "bgr": (249, 202, 144),
        "valence": "Neutral",
        "description": "Resting baseline expression, balanced emotional state."
    },
    "sad": {
        "label": "Sad",
        "emoji": "😢",
        "color": "#29B6F6",
        "bgr": (246, 182, 41),
        "valence": "Negative",
        "description": "Low arousal negative emotion, pulled down mouth corners."
    },
    "surprise": {
        "label": "Surprise",
        "emoji": "😲",
        "color": "#FF9100",
        "bgr": (0, 145, 255),
        "valence": "Positive",
        "description": "High arousal reaction to unexpected stimulus, dropped jaw."
    }
}


def load_emotion_model(model_path: str = "models/emotion_cnn.keras"):
    """Load and return the trained TensorFlow emotion model."""
    model = tf.keras.models.load_model(model_path)
    # Warm up inference with a dummy input
    dummy = np.zeros((1, 48, 48, 1), dtype=np.float32)
    model.predict(dummy, verbose=0)
    return model


def load_face_detector():
    """Load and return the OpenCV Haar Cascade face detector."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    return detector


def draw_cyber_hud(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    emotion: str,
    confidence: float,
    color_bgr: Tuple[int, int, int] = (0, 214, 255),
    face_idx: int = 1
) -> np.ndarray:
    """Draw a futuristic HUD bounding box with corner brackets and glowing label chip."""
    overlay = frame.copy()
    
    # 1. Subtle semi-transparent bounding fill
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color_bgr, -1)
    cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

    # 2. Main thin rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 1, cv2.LINE_AA)

    # 3. Corner Brackets (futuristic HUD)
    corner_len = max(12, int(min(w, h) * 0.18))
    thickness = 3

    # Top-Left
    cv2.line(frame, (x, y), (x + corner_len, y), color_bgr, thickness, cv2.LINE_AA)
    cv2.line(frame, (x, y), (x, y + corner_len), color_bgr, thickness, cv2.LINE_AA)

    # Top-Right
    cv2.line(frame, (x + w, y), (x + w - corner_len, y), color_bgr, thickness, cv2.LINE_AA)
    cv2.line(frame, (x + w, y), (x + w, y + corner_len), color_bgr, thickness, cv2.LINE_AA)

    # Bottom-Left
    cv2.line(frame, (x, y + h), (x + corner_len, y + h), color_bgr, thickness, cv2.LINE_AA)
    cv2.line(frame, (x, y + h), (x, y + h - corner_len), color_bgr, thickness, cv2.LINE_AA)

    # Bottom-Right
    cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color_bgr, thickness, cv2.LINE_AA)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color_bgr, thickness, cv2.LINE_AA)

    # 4. Floating Header Badge with Emotion & Confidence
    meta = EMOTION_META.get(emotion, {"label": emotion.capitalize(), "emoji": ""})
    label_text = f"#{face_idx} {meta['label'].upper()} {confidence:.1f}%"
    
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.55
    text_thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, text_thickness)

    badge_h = text_h + 16
    badge_w = text_w + 16
    badge_y1 = max(0, y - badge_h - 4)
    badge_y2 = badge_y1 + badge_h
    badge_x1 = x
    badge_x2 = x + badge_w

    # Dark background pill for badge
    badge_overlay = frame.copy()
    cv2.rectangle(badge_overlay, (badge_x1, badge_y1), (badge_x2, badge_y2), (15, 17, 23), -1)
    cv2.addWeighted(badge_overlay, 0.85, frame, 0.15, 0, frame)

    # Accent left border indicator
    cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x1 + 4, badge_y2), color_bgr, -1)
    cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), color_bgr, 1, cv2.LINE_AA)

    # Text rendering
    text_x = badge_x1 + 10
    text_y = badge_y1 + text_h + 8
    cv2.putText(frame, label_text, (text_x, text_y), font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

    return frame


def process_facial_emotions(
    img_bgr: np.ndarray,
    model: Any,
    face_detector: Any,
    scale_factor: float = 1.25,
    min_neighbors: int = 5,
    min_size: Tuple[int, int] = (30, 30)
) -> Dict[str, Any]:
    """
    Detects faces in an image and runs emotion recognition.
    Returns:
        dict containing:
            'annotated_bgr': image with HUD overlays
            'annotated_rgb': image in RGB format for display
            'faces': list of per-face analysis results
            'face_count': number of faces detected
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces_detected = face_detector.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size
    )

    annotated = img_bgr.copy()
    face_results = []

    for idx, (x, y, w, h) in enumerate(faces_detected, start=1):
        # Extract face region of interest
        face_roi = gray[y:y + h, x:x + w]
        if face_roi.size == 0:
            continue

        # Preprocess face to match model input (48x48, grayscale, normalized)
        resized_face = cv2.resize(face_roi, (48, 48))
        norm_face = resized_face.astype("float32") / 255.0
        norm_face = np.expand_dims(norm_face, axis=-1)
        norm_face = np.expand_dims(norm_face, axis=0)

        # Inference
        predictions = model.predict(norm_face, verbose=0)[0]
        emotion_idx = int(np.argmax(predictions))
        emotion = CLASS_NAMES[emotion_idx]
        confidence = float(predictions[emotion_idx]) * 100.0

        # Full probability distribution mapped to emotion names
        prob_dict = {
            CLASS_NAMES[i]: float(predictions[i]) * 100.0
            for i in range(len(CLASS_NAMES))
        }

        meta = EMOTION_META.get(emotion, {
            "label": emotion.capitalize(),
            "emoji": "😐",
            "color": "#90CAF9",
            "bgr": (255, 255, 255),
            "valence": "Neutral"
        })

        # Draw futuristic HUD
        draw_cyber_hud(
            annotated,
            x, y, w, h,
            emotion=emotion,
            confidence=confidence,
            color_bgr=meta["bgr"],
            face_idx=idx
        )

        # Crop color face for avatar display
        crop_bgr = img_bgr[max(0, y):min(img_bgr.shape[0], y + h), max(0, x):min(img_bgr.shape[1], x + w)]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB) if crop_bgr.size > 0 else None

        face_results.append({
            "face_id": idx,
            "bbox": (int(x), int(y), int(w), int(h)),
            "dominant_emotion": emotion,
            "confidence": confidence,
            "probabilities": prob_dict,
            "metadata": meta,
            "crop_rgb": crop_rgb
        })

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    return {
        "annotated_bgr": annotated,
        "annotated_rgb": annotated_rgb,
        "faces": face_results,
        "face_count": len(face_results)
    }
