import cv2
import csv
import os
import mediapipe as mp

# =============================
# MEDIAPIPE SETUP
# =============================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)


# =============================
# SIGN NAME
# =============================

label = input(
    "Enter sign name (example: HELLO): "
).strip().upper()

if not label:
    print("No sign name entered.")
    landmarker.close()
    exit()


# =============================
# DATA FILE
# =============================

os.makedirs("data", exist_ok=True)

file_path = "data/isl_landmarks.csv"

file_exists = os.path.exists(file_path)


# =============================
# CAMERA
# =============================

camera = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not camera.isOpened():
    print("Camera could not be opened.")
    landmarker.close()
    exit()

print()
print("Camera started!")
print("Show the sign:", label)
print("Press SPACE to save a sample.")
print("Press Q to quit.")


# =============================
# CSV FILE
# =============================

with open(
    file_path,
    "a",
    newline=""
) as file:

    writer = csv.writer(file)

    if not file_exists:

        header = ["label"]

        for i in range(21):
            header.extend([
                f"x{i}",
                f"y{i}",
                f"z{i}"
            ])

        writer.writerow(header)


    # =========================
    # CAMERA LOOP
    # =========================

    while True:

        success, frame = camera.read()

        if not success:
            print("Could not read camera.")
            break

        frame = cv2.flip(
            frame,
            1
        )

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(
            mp_image
        )

        # -------------------------
        # Hand detected
        # -------------------------

        if result.hand_landmarks:

            cv2.putText(
                frame,
                "HAND DETECTED",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "SHOW YOUR HAND",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )


        cv2.putText(
            frame,
            "SPACE = SAVE SAMPLE",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Q = QUIT",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        cv2.imshow(
            "ISL Dataset Collection",
            frame
        )


        # =========================
        # KEYBOARD
        # =========================

        key = cv2.waitKey(1) & 0xFF


        # Save sample
        if key == 32:

            if result.hand_landmarks:

                landmarks = result.hand_landmarks[0]

                row = [label]

                for landmark in landmarks:

                    row.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ])

                writer.writerow(row)

                file.flush()

                print(
                    f"Saved sample for {label}"
                )

            else:

                print(
                    "No hand detected. "
                    "Try again."
                )


        # Quit
        if key == ord("q"):

            break


# =============================
# CLEANUP
# =============================

camera.release()

landmarker.close()

cv2.destroyAllWindows()

print()
print("Dataset collection finished.")
print("Saved to:", file_path)