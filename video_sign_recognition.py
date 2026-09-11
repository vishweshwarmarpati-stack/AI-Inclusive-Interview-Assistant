import cv2
import numpy as np
import mediapipe as mp
import joblib
import time

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = joblib.load("models/sign_model.pkl")

# ==========================================
# MEDIAPIPE HAND DETECTION
# ==========================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================================
# START CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

# ==========================================
# SENTENCE VARIABLES
# ==========================================

sentence = []

# Last accepted sign
last_sign = None
last_time = 0

# Current sign being checked
candidate_sign = None
candidate_count = 0

# Number of frames required for confirmation
REQUIRED_FRAMES = 8

# Minimum time between two different signs
SIGN_DELAY = 1.5

# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera could not be opened.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand
    results = hands.process(rgb)

    current_sign = None

    # ==========================================
    # IF HAND IS DETECTED
    # ==========================================

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # ==========================================
            # GET 21 HAND LANDMARKS
            # ==========================================

            data = []

            for landmark in hand_landmarks.landmark:

                data.append(landmark.x)
                data.append(landmark.y)
                data.append(landmark.z)

            # Convert to model input
            input_data = np.array(data).reshape(1, -1)

            # ==========================================
            # PREDICT SIGN
            # ==========================================

            prediction = model.predict(input_data)

            current_sign = str(prediction[0])

            # Show current prediction
            cv2.putText(
                frame,
                "Current Sign: " + current_sign,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            # ==========================================
            # STABILIZE PREDICTION
            # ==========================================

            if current_sign == candidate_sign:

                candidate_count += 1

            else:

                candidate_sign = current_sign
                candidate_count = 1

    # ==========================================
    # ACCEPT STABLE SIGN
    # ==========================================

    current_time = time.time()

    if current_sign is not None:

        if (
            candidate_count >= REQUIRED_FRAMES
            and candidate_sign != last_sign
            and current_time - last_time >= SIGN_DELAY
        ):

            sentence.append(candidate_sign)

            last_sign = candidate_sign
            last_time = current_time

            candidate_count = 0

    # ==========================================
    # DISPLAY SENTENCE
    # ==========================================

    sentence_text = " ".join(sentence)

    cv2.putText(
        frame,
        "Sentence: " + sentence_text,
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # ==========================================
    # INSTRUCTIONS
    # ==========================================

    cv2.putText(
        frame,
        "C = Clear Sentence",
        (30, 430),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Q = Quit",
        (30, 460),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # ==========================================
    # SHOW CAMERA
    # ==========================================

    cv2.imshow(
        "AI Inclusive Interview Assistant",
        frame
    )

    # ==========================================
    # KEYBOARD CONTROLS
    # ==========================================

    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord("q"):
        break

    # Clear sentence
    if key == ord("c"):

        sentence = []
        last_sign = None
        last_time = 0
        candidate_sign = None
        candidate_count = 0

# ==========================================
# CLOSE EVERYTHING
# ==========================================

cap.release()
cv2.destroyAllWindows()
hands.close()