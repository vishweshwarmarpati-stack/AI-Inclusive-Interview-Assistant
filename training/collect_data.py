import cv2
import csv
import os
import mediapipe as mp

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "isl_landmarks_2hand.csv"
)

# --------------------------------------------------
# MEDIAPIPE HANDS
# --------------------------------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --------------------------------------------------
# CSV
# --------------------------------------------------

os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

file_exists = os.path.exists(DATA_PATH)

header = ["label"]

# Left hand = 21 landmarks × 3
for i in range(21):
    header.extend([
        f"left_x{i}",
        f"left_y{i}",
        f"left_z{i}"
    ])

# Right hand = 21 landmarks × 3
for i in range(21):
    header.extend([
        f"right_x{i}",
        f"right_y{i}",
        f"right_z{i}"
    ])

# --------------------------------------------------
# START
# --------------------------------------------------

label = input("Enter sign label: ").strip()

if not label:
    print("Label cannot be empty.")
    exit()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("\n====================================")
print("     TWO-HAND DATA COLLECTION")
print("====================================")
print(f"Sign: {label}")
print()
print("SPACE  -> Save sample")
print("Q      -> Quit")
print("====================================\n")

sample_count = 0

with open(DATA_PATH, "a", newline="") as file:

    writer = csv.writer(file)

    if not file_exists:
        writer.writerow(header)

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Could not read camera.")
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # BGR -> RGB
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # MediaPipe processing
        results = hands.process(rgb)

        # --------------------------------------------------
        # DEFAULT: NO HAND
        # --------------------------------------------------

        left_hand = [0.0] * 63
        right_hand = [0.0] * 63

        hands_detected = 0

        # --------------------------------------------------
        # EXTRACT HANDS
        # --------------------------------------------------

        if results.multi_hand_landmarks:

            hands_detected = len(
                results.multi_hand_landmarks
            )

            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):

                hand = []

                for landmark in hand_landmarks.landmark:

                    hand.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ])

                hand_type = handedness.classification[0].label

                if hand_type == "Left":
                    left_hand = hand

                elif hand_type == "Right":
                    right_hand = hand

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        # --------------------------------------------------
        # DISPLAY
        # --------------------------------------------------

        cv2.putText(
            frame,
            f"Sign: {label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Hands: {hands_detected}/2",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Samples: {sample_count}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "SPACE = Save | Q = Quit",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Two-Hand Sign Collection",
            frame
        )

        # --------------------------------------------------
        # KEYBOARD
        # --------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        # SPACE
        if key == 32:

            row = (
                [label]
                + left_hand
                + right_hand
            )

            writer.writerow(row)
            file.flush()

            sample_count += 1

            print(
                f"Saved sample {sample_count}"
            )

        # Q
        elif key == ord("q"):
            break

# --------------------------------------------------
# CLEANUP
# --------------------------------------------------

cap.release()
cv2.destroyAllWindows()
hands.close()

print("\n====================================")
print("Collection finished!")
print(f"Samples collected: {sample_count}")
print(f"Saved to: {DATA_PATH}")
print("====================================")