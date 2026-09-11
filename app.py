import streamlit as st
import cv2
import numpy as np
import joblib
import mediapipe as mp
import pyttsx3
import av
import time

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from faster_whisper import WhisperModel


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Inclusive Interview Assistant",
    page_icon="🤝",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "candidate_message" not in st.session_state:
    st.session_state["candidate_message"] = ""

if "final_response" not in st.session_state:
    st.session_state["final_response"] = ""

if "interviewer_question" not in st.session_state:
    st.session_state["interviewer_question"] = ""

if "simplified_question" not in st.session_state:
    st.session_state["simplified_question"] = ""


# ============================================================
# TITLE
# ============================================================

st.title("🤝 AI Inclusive Interview Assistant")

st.markdown(
    """
    **An AI-powered communication bridge for inclusive interviews.**

    This system helps candidates communicate using sign language
    and helps interviewers communicate using speech-to-text and
    simplified questions.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Choose a page:",
    [
        "🏠 Home",
        "⚙️ Accessibility Setup",
        "🎤 Live Interview",
        "🧪 Practice Interview"
    ]
)


# ============================================================
# SIGN VIDEO PROCESSOR
# ============================================================

class SignVideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.model = joblib.load(
            "models/sign_model.pkl"
        )

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.candidate_sign = None
        self.candidate_count = 0

        self.last_sign = None
        self.last_time = 0

        self.sentence = []

        self.REQUIRED_FRAMES = 8
        self.SIGN_DELAY = 1.5

    def recv(self, frame):

        image = frame.to_ndarray(
            format="bgr24"
        )

        image = cv2.flip(
            image,
            1
        )

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        results = self.hands.process(
            rgb
        )

        current_sign = None

        if results.multi_hand_landmarks:

            hand_landmarks = results.multi_hand_landmarks[0]

            self.mp_draw.draw_landmarks(
                image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )

            data = []

            for landmark in hand_landmarks.landmark:

                data.append(landmark.x)
                data.append(landmark.y)
                data.append(landmark.z)

            input_data = np.array(
                data
            ).reshape(
                1,
                -1
            )

            prediction = self.model.predict(
                input_data
            )

            current_sign = str(
                prediction[0]
            )

            if current_sign == self.candidate_sign:

                self.candidate_count += 1

            else:

                self.candidate_sign = current_sign
                self.candidate_count = 1

            current_time = time.time()

            if (
                self.candidate_count >= self.REQUIRED_FRAMES
                and self.candidate_sign != self.last_sign
                and current_time - self.last_time >= self.SIGN_DELAY
            ):

                self.sentence.append(
                    self.candidate_sign
                )

                self.last_sign = self.candidate_sign
                self.last_time = current_time
                self.candidate_count = 0

            cv2.putText(
                image,
                "Current Sign: " + current_sign,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                image,
                "SHOW YOUR HAND",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

        sentence_text = " ".join(
            self.sentence
        )

        cv2.putText(
            image,
            "Sentence: " + sentence_text,
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.header("🌟 Welcome")

    st.write(
        """
        The AI Inclusive Interview Assistant is designed to make
        interviews more accessible for Deaf, hard-of-hearing,
        and non-speaking candidates.
        """
    )

    st.subheader("🔄 How the system works")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### 🧑‍💻 Candidate → Interviewer"
        )

        st.write(
            """
            ✋ Candidate uses sign language

            ↓

            🤖 AI recognizes the sign

            ↓

            📝 Signs are converted into text

            ↓

            ✏️ Candidate can edit the response

            ↓

            🔊 AI speaks the response
            """
        )

    with col2:

        st.markdown(
            "### 🎤 Interviewer → Candidate"
        )

        st.write(
            """
            🎤 Interviewer speaks

            ↓

            🤖 Whisper converts speech to text

            ↓

            📝 Interview question appears

            ↓

            ✨ AI simplifies the question

            ↓

            👤 Candidate can understand and respond
            """
        )

    st.success(
        "The goal is to provide a communication bridge "
        "without requiring a human assistant during the interview."
    )


# ============================================================
# ACCESSIBILITY SETUP
# ============================================================

elif page == "⚙️ Accessibility Setup":

    st.header("⚙️ Accessibility Setup")

    st.write(
        "Select the accessibility features you want to use."
    )

    sign_enabled = st.checkbox(
        "✋ Sign Language Recognition",
        value=True
    )

    captions_enabled = st.checkbox(
        "📝 Speech-to-Text Captions",
        value=True
    )

    tts_enabled = st.checkbox(
        "🔊 Text-to-Speech",
        value=True
    )

    ai_enabled = st.checkbox(
        "🤖 AI Question Simplification",
        value=True
    )

    st.divider()

    st.subheader("Selected Features")

    if sign_enabled:
        st.write("✅ Sign Language Recognition")

    if captions_enabled:
        st.write("✅ Speech-to-Text")

    if tts_enabled:
        st.write("✅ Text-to-Speech")

    if ai_enabled:
        st.write("✅ AI Question Simplification")


# ============================================================
# LIVE INTERVIEW
# ============================================================

elif page == "🎤 Live Interview":

    st.header("🎤 Live Interview")

    st.write(
        "Two-way communication between the candidate and interviewer."
    )

    candidate_col, interviewer_col = st.columns(2)


    # ========================================================
    # CANDIDATE SIDE
    # ========================================================

    with candidate_col:

        st.subheader("🧑‍💻 Candidate")

        st.markdown(
            "### ✋ Sign Language Recognition"
        )

        st.write(
            "Show your sign to the camera."
        )

        ctx = webrtc_streamer(
            key="sign-recognition",
            video_processor_factory=SignVideoProcessor,
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_processing=True
        )

        st.divider()

        # ----------------------------------------------------
        # USE RECOGNIZED SENTENCE
        # ----------------------------------------------------

        if st.button(
            "⬇️ Use Recognized Sentence",
            key="use_recognized"
        ):

            if ctx.video_processor:

                recognized_sentence = " ".join(
                    ctx.video_processor.sentence
                )

                if recognized_sentence:

                    st.session_state[
                        "candidate_message"
                    ] = recognized_sentence

                    st.success(
                        "Recognized sentence added!"
                    )

                    st.rerun()

                else:

                    st.warning(
                        "No signs have been recognized yet."
                    )


        # ----------------------------------------------------
        # CLEAR RECOGNIZED SENTENCE
        # ----------------------------------------------------

        if st.button(
            "🗑️ Clear Recognized Sentence",
            key="clear_recognized"
        ):

            if ctx.video_processor:

                ctx.video_processor.sentence = []

                ctx.video_processor.last_sign = None

                ctx.video_processor.candidate_sign = None

                ctx.video_processor.candidate_count = 0

                ctx.video_processor.last_time = 0

                st.session_state[
                    "candidate_message"
                ] = ""

                st.session_state[
                    "final_response"
                ] = ""

                st.success(
                    "Sentence cleared."
                )

        st.divider()

        # ----------------------------------------------------
        # CANDIDATE RESPONSE
        # ----------------------------------------------------

        st.subheader(
            "📝 Candidate Response"
        )

        candidate_message = st.text_area(
            "✏️ Edit your response if needed:",
            value=st.session_state.get(
                "candidate_message",
                ""
            ),
            height=120,
            placeholder=(
                "Your recognized sign response "
                "will appear here..."
            )
        )

        # ----------------------------------------------------
        # ACCEPT RESPONSE
        # ----------------------------------------------------

        if st.button(
            "✅ Accept Response",
            key="accept_response"
        ):

            if candidate_message.strip():

                st.session_state[
                    "candidate_message"
                ] = candidate_message.strip()

                st.session_state[
                    "final_response"
                ] = candidate_message.strip()

                st.success(
                    "Response accepted!"
                )

            else:

                st.warning(
                    "Please enter or recognize a response first."
                )

        # ----------------------------------------------------
        # FINAL RESPONSE
        # ----------------------------------------------------

        if st.session_state.get(
            "final_response"
        ):

            st.subheader(
                "📋 Final Response"
            )

            st.info(
                st.session_state[
                    "final_response"
                ]
            )

        # ----------------------------------------------------
        # CLEAR RESPONSE
        # ----------------------------------------------------

        if st.button(
            "🗑️ Clear Response",
            key="clear_response"
        ):

            st.session_state[
                "candidate_message"
            ] = ""

            st.session_state[
                "final_response"
            ] = ""

            st.success(
                "Response cleared."
            )

            st.rerun()

        # ----------------------------------------------------
        # SPEAK RESPONSE
        # ----------------------------------------------------

        if st.button(
            "🔊 Speak Response",
            key="speak_response"
        ):

            response = st.session_state.get(
                "final_response",
                ""
            )

            if response.strip():

                try:

                    engine = pyttsx3.init()

                    engine.say(
                        response
                    )

                    engine.runAndWait()

                    engine.stop()

                    st.success(
                        "Response spoken successfully!"
                    )

                except Exception as e:

                    st.error(
                        f"Text-to-Speech error: {e}"
                    )

            else:

                st.warning(
                    "Please accept a response first."
                )


    # ========================================================
    # INTERVIEWER SIDE
    # ========================================================

    with interviewer_col:

        st.subheader("🎤 Interviewer")

        st.markdown(
            "### 🎙️ Ask a Question"
        )

        st.write(
            "Record your interview question."
        )

        audio_value = st.audio_input(
            "🎤 Record Question",
            key="interviewer_audio"
        )

        if audio_value:

            audio_file = "interviewer_audio.wav"

            with open(
                audio_file,
                "wb"
            ) as f:

                f.write(
                    audio_value.getbuffer()
                )

            st.audio(
                audio_value
            )

            # ------------------------------------------------
            # SPEECH TO TEXT
            # ------------------------------------------------

            if st.button(
                "📝 Convert Speech to Text",
                key="convert_speech"
            ):

                with st.spinner(
                    "Converting speech to text..."
                ):

                    try:

                        model = WhisperModel(
                            "tiny",
                            device="cpu",
                            compute_type="int8"
                        )

                        segments, info = model.transcribe(
                            audio_file
                        )

                        text = " ".join(
                            segment.text
                            for segment in segments
                        )

                        text = text.strip()

                        st.session_state[
                            "interviewer_question"
                        ] = text

                        st.success(
                            "Speech converted successfully!"
                        )

                    except Exception as e:

                        st.error(
                            f"Speech-to-text error: {e}"
                        )


        # ----------------------------------------------------
        # SHOW INTERVIEWER QUESTION
        # ----------------------------------------------------

        if st.session_state.get(
            "interviewer_question"
        ):

            st.subheader(
                "📝 Interviewer Question"
            )

            st.info(
                st.session_state[
                    "interviewer_question"
                ]
            )

            # ------------------------------------------------
            # SIMPLIFY QUESTION
            # ------------------------------------------------

            if st.button(
                "✨ Simplify Question",
                key="simplify_question"
            ):

                question = st.session_state[
                    "interviewer_question"
                ].strip()

                q = question.lower().strip(
                    " ?.! "
                )

                simplifications = {

                    "tell me about yourself":
                        "Please introduce yourself and briefly talk about your education, skills and interests.",

                    "tell me about your family":
                        "Please briefly describe your family.",

                    "what are your strengths":
                        "Please tell me what you are good at.",

                    "what are your weaknesses":
                        "Please tell me something you want to improve.",

                    "why should we hire you":
                        "Please explain why you are suitable for this job.",

                    "why do you want this job":
                        "Please explain why you are interested in this job.",

                    "why do you want to work here":
                        "Please explain why you want to work at this company.",

                    "where do you see yourself in five years":
                        "Please describe your future career goals.",

                    "what are your career goals":
                        "Please tell me about your future career plans.",

                    "why did you choose this field":
                        "Please explain why you chose this field of study or work.",

                    "what do you know about our company":
                        "Please tell me what you know about this company.",

                    "why should we select you":
                        "Please explain why you are a good choice for this position.",

                    "tell me about your education":
                        "Please briefly describe your education.",

                    "tell me about your skills":
                        "Please describe the skills you have.",

                    "what is your biggest achievement":
                        "Please describe something important that you have achieved.",

                    "do you have any questions for us":
                        "Please tell us if you have any questions."
                }

                if q in simplifications:

                    simplified_question = simplifications[q]

                else:

                    simplified_question = question

                    for original, simple in simplifications.items():

                        if original in q:

                            simplified_question = simple

                            break

                st.session_state[
                    "simplified_question"
                ] = simplified_question

                st.success(
                    "Question simplified successfully!"
                )

            # ------------------------------------------------
            # SHOW SIMPLIFIED QUESTION
            # ------------------------------------------------

            if st.session_state.get(
                "simplified_question"
            ):

                st.subheader(
                    "✨ Simplified Question"
                )

                st.success(
                    st.session_state[
                        "simplified_question"
                    ]
                )

        else:

            st.info(
                "Record and convert a question to see it here."
            )


# ============================================================
# PRACTICE INTERVIEW
# ============================================================

elif page == "🧪 Practice Interview":

    st.header("🧪 Practice Interview")

    st.write(
        "Practice answering common interview questions."
    )

    questions = [
        "Tell me about yourself.",
        "Tell me about your family.",
        "What are your strengths?",
        "What are your weaknesses?",
        "Why should we hire you?"
    ]

    for i, question in enumerate(
        questions,
        start=1
    ):

        st.subheader(
            f"Question {i}"
        )

        st.write(
            question
        )

        st.text_area(
            "Your answer:",
            key=f"practice_{i}",
            height=100
        )

        st.divider()


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.info(
    """
    🤝 AI Inclusive Interview Assistant

    Making interviews more accessible through AI.
    """
)