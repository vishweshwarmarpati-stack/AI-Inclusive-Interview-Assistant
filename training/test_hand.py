import mediapipe as mp

print("MediaPipe version:", mp.__version__)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE
)

print("Loading hand model...")

with HandLandmarker.create_from_options(options) as landmarker:
    print("Hand Landmarker loaded successfully!")