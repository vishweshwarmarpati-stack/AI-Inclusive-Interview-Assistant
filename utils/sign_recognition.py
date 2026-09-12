import os
import cv2
import mediapipe as mp
import joblib
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "data",
    "models",
    "sign_model_2hand.pkl"
)

model = joblib.load(MODEL_PATH)

mp_hands = mp.solutions.hands


hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def recognize_sign(frame):

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if not results.multi_hand_landmarks:
        return None

    left_hand = np.zeros(63)
    right_hand = np.zeros(63)

    for hand_landmarks, handedness in zip(
        results.multi_hand_landmarks,
        results.multi_handedness
    ):

        landmarks = []

        for landmark in hand_landmarks.landmark:
            landmarks.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        landmarks = np.array(landmarks)

        label = handedness.classification[0].label

        if label == "Left":
            left_hand = landmarks

        elif label == "Right":
            right_hand = landmarks

    features = np.concatenate([
        left_hand,
        right_hand
    ])

    features = features.reshape(1, -1)

    prediction = model.predict(features)

    return prediction[0]
