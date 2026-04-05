"""
Real-Time Emotion Detection using Webcam
=========================================
Loads the trained CNN-BiLSTM model and performs real-time
emotion detection via webcam feed.

Requirements:
    pip install tensorflow opencv-python numpy

Usage:
    python realtime_detection.py --model best_emotion_model.keras
"""

import cv2
import numpy as np
import argparse
import tensorflow as tf

EMOTION_LABELS = [
    'Surprise', 'Fear', 'Disgust', 'Happiness',
    'Sadness', 'Anger', 'Neutral'
]

EMOTION_COLORS = {
    'Surprise': (0, 255, 255),
    'Fear': (128, 0, 128),
    'Disgust': (0, 128, 0),
    'Happiness': (0, 255, 0),
    'Sadness': (255, 0, 0),
    'Anger': (0, 0, 255),
    'Neutral': (200, 200, 200)
}

CAREGIVER_SUGGESTIONS = {
    'Happiness': [
        "The individual appears happy! Reinforce this positive emotion.",
        "Engage in shared activities to build social connection.",
        "Use this moment for learning — positive emotions aid retention."
    ],
    'Sadness': [
        "The individual may be feeling sad. Offer comfort gently.",
        "Use calming sensory tools (weighted blanket, soft music).",
        "Avoid overstimulation; provide a quiet, safe space."
    ],
    'Anger': [
        "Signs of frustration detected. Stay calm and patient.",
        "Remove potential triggers from the environment.",
        "Offer a break or redirect to a calming activity."
    ],
    'Fear': [
        "The individual may be anxious. Provide reassurance.",
        "Use visual schedules to reduce uncertainty.",
        "Speak softly and maintain a non-threatening posture."
    ],
    'Surprise': [
        "Unexpected stimuli detected. Check for environmental changes.",
        "Guide the individual through the new experience gently.",
        "Use social stories to explain unexpected events."
    ],
    'Disgust': [
        "Possible sensory discomfort detected.",
        "Check for overwhelming smells, textures, or tastes.",
        "Offer sensory alternatives that the individual prefers."
    ],
    'Neutral': [
        "The individual appears calm and regulated.",
        "Good time for structured learning activities.",
        "Maintain the current environment and routine."
    ]
}

IMG_SIZE = 40


def main(args):
    # Load model
    print(f"Loading model from {args.model}...")
    model = tf.keras.models.load_model(args.model)
    print("Model loaded successfully!")

    # Face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Press 'q' to quit, 's' to show suggestions")
    current_emotion = 'Neutral'
    show_suggestions = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
            face_normalized = face_resized.astype('float32') / 255.0
            face_input = np.expand_dims(face_normalized, axis=0)

            predictions = model.predict(face_input, verbose=0)
            emotion_idx = np.argmax(predictions[0])
            confidence = predictions[0][emotion_idx]
            current_emotion = EMOTION_LABELS[emotion_idx]

            color = EMOTION_COLORS.get(current_emotion, (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label = f"{current_emotion} ({confidence:.1%})"
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Display info bar
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (50, 50, 50), -1)
        cv2.putText(frame, f"Detected Emotion: {current_emotion}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show suggestions overlay
        if show_suggestions:
            overlay = frame.copy()
            h_frame = frame.shape[0]
            cv2.rectangle(overlay, (10, h_frame-180), (500, h_frame-10), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            suggestions = CAREGIVER_SUGGESTIONS.get(current_emotion, [])
            cv2.putText(frame, "Caregiver Suggestions:", (20, h_frame-155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            for i, suggestion in enumerate(suggestions[:3]):
                y_pos = h_frame - 130 + i * 40
                text = suggestion[:60] + "..." if len(suggestion) > 60 else suggestion
                cv2.putText(frame, f"• {text}", (20, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.imshow('ASD Emotion Recognition System', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            show_suggestions = not show_suggestions

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-Time Emotion Detection')
    parser.add_argument('--model', type=str, default='best_emotion_model.keras',
                        help='Path to trained model file')
    args = parser.parse_args()
    main(args)
