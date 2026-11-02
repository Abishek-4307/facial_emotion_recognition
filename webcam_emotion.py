import cv2
import numpy as np
import tensorflow as tf

# Load trained model
MODEL_PATH = "models/emotion_cnn.keras"

model = tf.keras.models.load_model(MODEL_PATH)

# Emotion labels must match the class order used during training
class_names = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# Load OpenCV face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

print("Webcam started. Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read webcam frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]

        # Preprocess face to match training input
        face = cv2.resize(face, (48, 48))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        # Predict emotion
        predictions = model.predict(face, verbose=0)

        emotion_index = int(np.argmax(predictions[0]))
        emotion = class_names[emotion_index]
        confidence = float(predictions[0][emotion_index]) * 100

        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Display emotion and confidence
        label = f"{emotion}: {confidence:.1f}%"

        cv2.putText(
            frame,
            label,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("Facial Emotion Recognition", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows