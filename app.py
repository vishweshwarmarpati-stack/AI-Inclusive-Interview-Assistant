import streamlit as st
import cv2
import numpy as np
import joblib
import mediapipe as mp
import pyttsx3
import av
import time
import tempfile
import os

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from faster_whisper import WhisperModel


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Inclusive Interview Assistant",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM FRONTEND CSS
# =========================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* SIDEBAR */

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,0.20);
    }

    .sidebar-logo {
        text-align: center;
        font-size: 45px;
        margin-bottom: 0px;
    }

    .sidebar-title {
        text-align: center;
        font-size: 20px;
        font-weight: 800;
    }

    .sidebar-subtitle {
        text-align: center;
        font-size: 12px;
        opacity: 0.65;
        margin-bottom: 30px;
    }

    .goto-title {
        font-size: 27px;
        font-weight: 900;
        margin-bottom: 12px;
    }

    /* HERO */

    .hero {
        padding: 38px;
        border-radius: 25px;
        border: 1px solid rgba(128,128,128,0.20);
        background: linear-gradient(
            135deg,
            rgba(99,102,241,0.13),
            rgba(168,85,247,0.08)
        );
        margin-bottom: 30px;
    }

    .hero-badge {
        display: inline-block;
        padding: 7px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 800;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 15px;
    }

    .hero-title {
        font-size: 44px;
        line-height: 1.1;
        font-weight: 900;
        margin-bottom: 15px;
    }

    .hero-text {
        font-size: 17px;
        line-height: 1.65;
        opacity: 0.75;
        max-width: 900px;
    }

    /* FEATURE CARDS */

    .feature-card {
        padding: 25px;
        border-radius: 20px;
        border: 1px solid rgba(128,128,128,0.20);
        min-height: 180px;
        margin-bottom: 18px;
    }

    .feature-icon {
        font-size: 38px;
        margin-bottom: 10px;
    }

    .feature-title {
        font-size: 20px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .feature-description {
        font-size: 14px;
        line-height: 1.6;
        opacity: 0.7;
    }

    /* SECTION */

    .section-title {
        font-size: 30px;
        font-weight: 850;
        margin-bottom: 5px;
    }

    .section-subtitle {
        opacity: 0.65;
        margin-bottom: 22px;
    }

    /* TOOL CARDS */

    .tool-card {
        padding: 22px;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,0.20);
        margin-bottom: 15px;
    }

    .tool-card-title {
        font-size: 21px;
        font-weight: 800;
    }

    /* FOOTER */

    .footer {
        text-align: center;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid rgba(128,128,128,0.15);
        opacity: 0.55;
        font-size: 13px;
    }

    /* BUTTONS */

    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
        min-height: 45px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TEXT TO SPEECH
# =========================================================

def generate_speech(text):

    if not text.strip():
        return None

    try:

        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".aiff"
        )

        audio_path = temp_file.name

        temp_file.close()

        engine = pyttsx3.init()

        engine.setProperty(
            "rate",
            160
        )

        engine.save_to_file(
            text,
            audio_path
        )

        engine.runAndWait()

        engine.stop()

        return audio_path

    except Exception as e:

        st.error(
            f"Text-to-Speech error: {e}"
        )

        return None


# =========================================================
# SPEECH TO TEXT
# =========================================================

@st.cache_resource
def load_whisper_model():

    return WhisperModel(
        "base",
        device="cpu",
        compute_type="int8"
    )


def transcribe_audio(audio_file):

    try:

        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        )

        temp_file.write(
            audio_file.getvalue()
        )

        audio_path = temp_file.name

        temp_file.close()

        model = load_whisper_model()

        segments, info = model.transcribe(
            audio_path,
            beam_size=5
        )

        text = " ".join(
            segment.text.strip()
            for segment in segments
        )

        os.remove(audio_path)

        return text.strip()

    except Exception as e:

        st.error(
            f"Speech-to-Text error: {e}"
        )

        return ""


# =========================================================
# SIGN LANGUAGE VIDEO PROCESSOR
# =========================================================

class SignVideoProcessor(VideoProcessorBase):

    def __init__(self):

        # Load two-hand model

        self.model = joblib.load(
            "data/models/sign_model_2hand.pkl"
        )

        # MediaPipe

        self.mp_hands = mp.solutions.hands

        self.mp_draw = (
            mp.solutions.drawing_utils
        )

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Stabilization

        self.candidate_sign = None

        self.candidate_count = 0

        self.last_sign = None

        self.last_time = 0

        self.sentence = []

        self.REQUIRED_FRAMES = 8

        self.SIGN_DELAY = 1.5


    # =====================================================
    # PROCESS FRAME
    # =====================================================

    def recv(self, frame):

        image = frame.to_ndarray(
            format="bgr24"
        )

        # Mirror camera

        image = cv2.flip(
            image,
            1
        )

        # BGR -> RGB

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # Detect hands

        results = self.hands.process(
            rgb
        )

        current_sign = None


        # =================================================
        # TWO HAND PROCESSING
        # =================================================

        if results.multi_hand_landmarks:

            left_hand = np.zeros(63)

            right_hand = np.zeros(63)


            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):

                # Draw landmarks

                self.mp_draw.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

                # Extract landmarks

                data = []

                for landmark in hand_landmarks.landmark:

                    data.append(
                        landmark.x
                    )

                    data.append(
                        landmark.y
                    )

                    data.append(
                        landmark.z
                    )

                data = np.array(data)

                # Identify hand

                hand_label = (
                    handedness
                    .classification[0]
                    .label
                )

                if hand_label == "Left":

                    left_hand = data

                elif hand_label == "Right":

                    right_hand = data


            # Combine both hands

            input_data = np.concatenate(
                [
                    left_hand,
                    right_hand
                ]
            ).reshape(
                1,
                -1
            )


            # Prediction

            prediction = self.model.predict(
                input_data
            )

            current_sign = str(
                prediction[0]
            )


            # =================================================
            # STABILIZATION
            # =================================================

            if current_sign == self.candidate_sign:

                self.candidate_count += 1

            else:

                self.candidate_sign = current_sign

                self.candidate_count = 1


            current_time = time.time()


            # Add stable sign

            if (
                self.candidate_count
                >= self.REQUIRED_FRAMES

                and current_sign
                != self.last_sign

                and current_time
                - self.last_time
                > self.SIGN_DELAY
            ):

                self.sentence.append(
                    current_sign
                )

                self.last_sign = (
                    current_sign
                )

                self.last_time = (
                    current_time
                )


        # =================================================
        # CURRENT SIGN
        # =================================================

        if current_sign:

            cv2.putText(
                image,
                f"Sign: {current_sign}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )


        # =================================================
        # SENTENCE
        # =================================================

        sentence_text = " ".join(
            self.sentence
        )

        cv2.putText(
            image,
            f"Sentence: {sentence_text}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    '<div class="sidebar-logo">🤝</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<div class="sidebar-title">AI Inclusive</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<div class="sidebar-subtitle">'
    'Interview Assistant'
    '</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<div class="goto-title">🚀 GO TO</div>',
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "",
    [
        "🏠 Home",
        "🎤 Live Interview",
        "📝 Interview Practice",
        "ℹ️ About"
    ],
    label_visibility="collapsed"
)


# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.markdown(
        """
        <div class="hero">

        <div class="hero-badge">
        ✨ AI-POWERED ACCESSIBILITY
        </div>

        <div class="hero-title">
        Breaking Barriers.<br>
        Building Inclusive Interviews.
        </div>

        <div class="hero-text">
        An intelligent interview assistant combining
        two-hand sign language recognition, speech
        recognition and text-to-speech technology
        to create a more accessible interview experience.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="section-title">'
        'Powerful Features'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Everything you need for an accessible interview.'
        '</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown(
            """
            <div class="feature-card">

            <div class="feature-icon">🤟</div>

            <div class="feature-title">
            Two-Hand Sign Recognition
            </div>

            <div class="feature-description">
            Recognize sign language gestures using
            both hands through a live camera feed.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            """
            <div class="feature-card">

            <div class="feature-icon">🎤</div>

            <div class="feature-title">
            Speech-to-Text
            </div>

            <div class="feature-description">
            Convert spoken answers into text using
            Faster-Whisper speech recognition.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            """
            <div class="feature-card">

            <div class="feature-icon">🔊</div>

            <div class="feature-title">
            Text-to-Speech
            </div>

            <div class="feature-description">
            Convert written responses into speech
            for easier communication.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("<br>", unsafe_allow_html=True)


    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            """
            <div class="feature-card">

            <div class="feature-icon">📝</div>

            <div class="feature-title">
            Interview Practice
            </div>

            <div class="feature-description">
            Practice common interview questions and
            receive instant feedback on your answers.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            """
            <div class="feature-card">

            <div class="feature-icon">♿</div>

            <div class="feature-title">
            Accessibility First
            </div>

            <div class="feature-description">
            Designed to reduce communication barriers
            in interview environments.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# LIVE INTERVIEW
# =========================================================

elif page == "🎤 Live Interview":

    st.markdown(
        '<div class="section-title">'
        '🎤 Live Interview Assistant'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Use Sign Language, Speech-to-Text or Text-to-Speech.'
        '</div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # SIGN LANGUAGE
    # =====================================================

    st.markdown(
        """
        <div class="tool-card">

        <div class="tool-card-title">
        🤟 Two-Hand Sign Language Recognition
        </div>

        <p>
        Allow camera access and perform a sign
        using one or both hands.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    webrtc_streamer(
        key="sign-language",
        video_processor_factory=SignVideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    # =====================================================
    # STT + TTS
    # =====================================================

    col1, col2 = st.columns(2)


    # =====================================================
    # SPEECH TO TEXT
    # =====================================================

    with col1:

        st.markdown(
            """
            <div class="tool-card">

            <div class="tool-card-title">
            🎤 Speech → Text
            </div>

            <p>
            Record your answer and convert your
            speech into text using Faster-Whisper.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


        audio_file = None


        if hasattr(
            st,
            "audio_input"
        ):

            audio_file = st.audio_input(
                "🎙️ Record your answer"
            )

        else:

            audio_file = st.file_uploader(
                "Upload an audio recording",
                type=[
                    "wav",
                    "mp3",
                    "m4a"
                ]
            )


        if audio_file is not None:

            if st.button(
                "📝 Convert Speech to Text",
                use_container_width=True
            ):

                with st.spinner(
                    "AI is listening..."
                ):

                    transcript = transcribe_audio(
                        audio_file
                    )


                if transcript:

                    st.session_state[
                        "transcript"
                    ] = transcript

                    st.success(
                        "Transcription complete! ✅"
                    )


        transcript = st.session_state.get(
            "transcript",
            ""
        )


        if transcript:

            st.text_area(
                "Your Transcript",
                value=transcript,
                height=150
            )


            if st.button(
                "🔊 Speak Transcript",
                use_container_width=True
            ):

                audio_path = generate_speech(
                    transcript
                )

                if audio_path:

                    st.audio(
                        audio_path
                    )


    # =====================================================
    # TEXT TO SPEECH
    # =====================================================

    with col2:

        st.markdown(
            """
            <div class="tool-card">

            <div class="tool-card-title">
            🔊 Text → Speech
            </div>

            <p>
            Enter text and convert it into
            computer-generated speech.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


        tts_text = st.text_area(
            "Enter text to speak",
            height=150,
            placeholder=(
                "Type your message here..."
            )
        )


        if st.button(
            "🔊 Generate Speech",
            use_container_width=True
        ):

            if tts_text.strip():

                with st.spinner(
                    "Generating voice..."
                ):

                    audio_path = generate_speech(
                        tts_text
                    )


                if audio_path:

                    st.success(
                        "Speech generated! 🔊"
                    )

                    st.audio(
                        audio_path
                    )

            else:

                st.warning(
                    "Please enter some text."
                )


# =========================================================
# INTERVIEW PRACTICE
# =========================================================

elif page == "📝 Interview Practice":

    st.markdown(
        '<div class="section-title">'
        '📝 Interview Practice'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Practice common interview questions and receive '
        'instant feedback on your answer.'
        '</div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # QUESTION
    # =====================================================

    question = st.selectbox(
        "🎯 Choose an interview question",
        [
            "Tell me about yourself.",
            "What are your strengths?",
            "What are your weaknesses?",
            "Why should we hire you?",
            "Where do you see yourself in five years?"
        ]
    )


    st.markdown(
        """
        <div class="tool-card">

        <div class="tool-card-title">
        💬 Interview Question
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.info(
        question
    )


    # =====================================================
    # ANSWER
    # =====================================================

    answer = st.text_area(
        "✍️ Your Answer",
        height=220,
        placeholder=(
            "Write your interview answer here..."
        )
    )


    # =====================================================
    # BUTTONS
    # =====================================================

    col1, col2 = st.columns(2)


    with col1:

        evaluate = st.button(
            "🧠 Evaluate Answer",
            use_container_width=True
        )


    with col2:

        speak = st.button(
            "🔊 Speak Answer",
            use_container_width=True
        )


    # =====================================================
    # SPEAK ANSWER
    # =====================================================

    if speak:

        if answer.strip():

            with st.spinner(
                "Generating speech..."
            ):

                audio_path = generate_speech(
                    answer
                )


            if audio_path:

                st.success(
                    "Speech generated successfully! 🔊"
                )

                st.audio(
                    audio_path
                )

        else:

            st.warning(
                "Please enter an answer first."
            )


    # =====================================================
    # EVALUATE ANSWER
    # =====================================================

    if evaluate:

        if not answer.strip():

            st.warning(
                "Please enter an answer before evaluating."
            )

        else:

            # -------------------------------------------------
            # BASIC TEXT ANALYSIS
            # -------------------------------------------------

            words = answer.split()

            word_count = len(words)


            sentences = [
                s.strip()
                for s in answer.replace(
                    "!",
                    "."
                ).replace(
                    "?",
                    "."
                ).split(".")
                if s.strip()
            ]


            sentence_count = max(
                len(sentences),
                1
            )


            # =================================================
            # CONTENT SCORE
            # =================================================

            if word_count >= 80:

                content_score = 9

            elif word_count >= 50:

                content_score = 8

            elif word_count >= 30:

                content_score = 7

            elif word_count >= 15:

                content_score = 6

            else:

                content_score = 4


            # =================================================
            # CLARITY SCORE
            # =================================================

            average_sentence_length = (
                word_count /
                sentence_count
            )


            if 8 <= average_sentence_length <= 25:

                clarity_score = 9

            elif 5 <= average_sentence_length <= 30:

                clarity_score = 8

            elif average_sentence_length <= 40:

                clarity_score = 7

            else:

                clarity_score = 5


            # =================================================
            # STRUCTURE SCORE
            # =================================================

            structure_words = [
                "because",
                "therefore",
                "however",
                "first",
                "second",
                "finally",
                "also",
                "experience",
                "example",
                "result"
            ]


            lower_answer = answer.lower()


            structure_matches = sum(
                1
                for word in structure_words
                if word in lower_answer
            )


            if structure_matches >= 3:

                structure_score = 9

            elif structure_matches == 2:

                structure_score = 8

            elif structure_matches == 1:

                structure_score = 7

            else:

                structure_score = 6


            # =================================================
            # RELEVANCE SCORE
            # =================================================

            if "strength" in question.lower():

                keywords = [
                    "skill",
                    "ability",
                    "experience",
                    "good",
                    "strength"
                ]


            elif "weakness" in question.lower():

                keywords = [
                    "weakness",
                    "improve",
                    "learning",
                    "challenge",
                    "development"
                ]


            elif "hire" in question.lower():

                keywords = [
                    "skill",
                    "experience",
                    "value",
                    "contribute",
                    "company"
                ]


            elif "five years" in question.lower():

                keywords = [
                    "future",
                    "career",
                    "goal",
                    "learn",
                    "grow"
                ]


            else:

                keywords = [
                    "student",
                    "experience",
                    "skill",
                    "project",
                    "interest"
                ]


            keyword_matches = sum(
                1
                for keyword in keywords
                if keyword in lower_answer
            )


            relevance_score = min(
                5 + keyword_matches,
                10
            )


            # =================================================
            # OVERALL SCORE
            # =================================================

            overall_score = round(
                (
                    relevance_score
                    + content_score
                    + clarity_score
                    + structure_score
                ) / 4,
                1
            )


            # =================================================
            # RESULTS
            # =================================================

            st.markdown(
                "<br>",
                unsafe_allow_html=True
            )


            st.markdown(
                '<div class="section-title">'
                '📊 Interview Evaluation'
                '</div>',
                unsafe_allow_html=True
            )


            if overall_score >= 8:

                st.success(
                    "🌟 Excellent answer! "
                    "You are showing strong interview preparation."
                )

            elif overall_score >= 6:

                st.info(
                    "👍 Good answer! "
                    "A few improvements can make it stronger."
                )

            else:

                st.warning(
                    "💪 Keep practicing! "
                    "Your answer needs more detail and structure."
                )


            # =================================================
            # SCORE BREAKDOWN
            # =================================================

            col1, col2, col3, col4 = st.columns(4)


            with col1:

                st.metric(
                    "🎯 Relevance",
                    f"{relevance_score}/10"
                )


            with col2:

                st.metric(
                    "📝 Content",
                    f"{content_score}/10"
                )


            with col3:

                st.metric(
                    "💬 Clarity",
                    f"{clarity_score}/10"
                )


            with col4:

                st.metric(
                    "🏗️ Structure",
                    f"{structure_score}/10"
                )


            # =================================================
            # OVERALL SCORE
            # =================================================

            st.markdown(
                "<br>",
                unsafe_allow_html=True
            )


            st.subheader(
                f"🏆 Overall Score: {overall_score}/10"
            )


            st.progress(
                min(
                    overall_score / 10,
                    1.0
                )
            )


            # =================================================
            # ANSWER STATISTICS
            # =================================================

            st.markdown(
                "<br>",
                unsafe_allow_html=True
            )


            st.subheader(
                "📈 Answer Statistics"
            )


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Words",
                    word_count
                )


            with col2:

                st.metric(
                    "Sentences",
                    sentence_count
                )


            with col3:

                st.metric(
                    "Avg. Words / Sentence",
                    round(
                        average_sentence_length,
                        1
                    )
                )


            # =================================================
            # FEEDBACK
            # =================================================

            st.markdown(
                "<br>",
                unsafe_allow_html=True
            )


            st.subheader(
                "💡 Personalized Feedback"
            )


            feedback = []


            if word_count < 30:

                feedback.append(
                    "📝 Your answer is too short. "
                    "Add specific examples, experiences or achievements."
                )

            elif word_count > 150:

                feedback.append(
                    "✂️ Your answer is quite long. "
                    "Try to make your response more concise."
                )

            else:

                feedback.append(
                    "✅ Your answer has a reasonable length."
                )


            if clarity_score < 7:

                feedback.append(
                    "💬 Try using shorter and clearer sentences."
                )

            else:

                feedback.append(
                    "✅ Your answer is reasonably clear."
                )


            if structure_score < 7:

                feedback.append(
                    "🏗️ Improve your structure by explaining "
                    "your point, giving an example and ending "
                    "with the result."
                )

            else:

                feedback.append(
                    "✅ Your answer shows a decent structure."
                )


            if relevance_score < 7:

                feedback.append(
                    "🎯 Focus more directly on the question "
                    "and include relevant examples."
                )

            else:

                feedback.append(
                    "✅ Your answer appears relevant."
                )


            for item in feedback:

                st.write(
                    item
                )


            # =================================================
            # INTERVIEW TIP
            # =================================================

            st.markdown(
                "<br>",
                unsafe_allow_html=True
            )


            st.subheader(
                "⭐ Interview Tip"
            )


            st.info(
                "A strong interview answer should usually "
                "follow this pattern: make your point → "
                "give a specific example → explain the result "
                "or what you learned."
            )


            st.caption(
                "This is a local rule-based evaluation system "
                "for interview practice. It provides general "
                "feedback and is not a professional hiring assessment."
            )


# =========================================================
# ABOUT
# =========================================================

elif page == "ℹ️ About":

    st.markdown(
        '<div class="section-title">'
        'ℹ️ About the Project'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="section-subtitle">'
        'Technology behind the AI Inclusive Interview Assistant.'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="hero">

        <div class="hero-title">
        🤝 AI Inclusive Interview Assistant
        </div>

        <div class="hero-text">

        This project aims to make interview environments
        more accessible by combining Artificial Intelligence
        with multiple communication technologies.

        The system supports two-hand sign language recognition,
        speech-to-text conversion, text-to-speech generation
        and interview practice.

        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="section-title">'
        '🛠️ Technology Stack'
        '</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown(
            """
            <div class="feature-card">

            <div class="feature-icon">🤟</div>

            <div class="feature-title">
            MediaPipe
            </div>

            <div class="feature-description">
            Detects hand landmarks and tracks
            two hands in real time.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            """
            <div class="feature-card">

            <div class="feature-icon">🌲</div>

            <div class="feature-title">
            Scikit-learn
            </div>

            <div class="feature-description">
            Machine learning model used for
            sign classification.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            """
            <div class="feature-card">

            <div class="feature-icon">🎤</div>

            <div class="feature-title">
            Faster-Whisper
            </div>

            <div class="feature-description">
            Converts spoken language into
            text using speech recognition.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown(
            """
            <div class="feature-card">

            <div class="feature-icon">🔊</div>

            <div class="feature-title">
            pyttsx3
            </div>

            <div class="feature-description">
            Converts text into computer-generated
            speech.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            """
            <div class="feature-card">

            <div class="feature-icon">🎥</div>

            <div class="feature-title">
            OpenCV
            </div>

            <div class="feature-description">
            Handles camera frames and computer
            vision processing.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            """
            <div class="feature-card">

            <div class="feature-icon">🌐</div>

            <div class="feature-title">
            Streamlit
            </div>

            <div class="feature-description">
            Provides the interactive web
            application interface.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

    🤝 AI Inclusive Interview Assistant
    &nbsp; • &nbsp;
    AI Powered
    &nbsp; • &nbsp;
    Accessibility First

    </div>
    """,
    unsafe_allow_html=True
)
