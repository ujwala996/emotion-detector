import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("emotion_model.hdf5")

emotion_labels = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

# Load face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48,48))
        face = face / 255.0
        face = np.reshape(face, (1,48,48,1))

        preds = model.predict(face)
        emotion = emotion_labels[np.argmax(preds)]

        # Draw rectangle
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

        # Emotion label
        cv2.putText(frame, emotion, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # Confidence display
        for i, (label, prob) in enumerate(zip(emotion_labels, preds[0])):
            text = f"{label}: {prob*100:.1f}%"
            cv2.putText(frame, text, (10, 30 + i*25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()