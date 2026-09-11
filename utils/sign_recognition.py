import cv2
import numpy as np
import mediapipe as mp
import joblib

# =========================
# LOAD TRAINED MODEL
# =========================

model = joblib.load("models/sign_model.pkl")

# =========================
# MEDIAPIPE HANDS
# =========================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# START CAMERA
# =========================

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw hand
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # =========================
            # EXTRACT LANDMARKS
            # =========================

            data = []

            for landmark in hand_landmarks.landmark:

                data.append(landmark.x)
                data.append(landmark.y)
                data.append(landmark.z)

            # Convert to NumPy array
            input_data = np.array(data).reshape(1, -1)

            # =========================
            # PREDICT SIGN
            # =========================

            prediction = model.predict(input_data)

            sign = prediction[0]

            # =========================
            # DISPLAY SIGN
            # =========================

            cv2.putText(
                frame,
                str(sign),
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3
            )

    # Show camera
    cv2.imshow(
        "AI Sign Recognition",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================
# CLOSE
# =========================

cap.release()
cv2.destroyAllWindows()