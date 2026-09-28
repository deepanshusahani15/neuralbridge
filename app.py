"""Neuralbridge three-mode communication MVP."""
from collections import deque
from pathlib import Path
import threading

import numpy as np
import streamlit as st

from backend.phrase_mapper import load_phrases
from backend.sign_recognition import CLASSIFIER, LANDMARK_MODEL, HandExtractor, load_classifier, predict
from backend.speech_to_text import transcribe
from backend.text_to_speech import synthesize
from backend.voice_response import select_response

st.set_page_config(page_title="Neuralbridge", page_icon="🫶", layout="wide")
st.title("Neuralbridge")
st.caption("A communication bridge for selected gestures, speech, and audio-first interaction")


@st.cache_resource
def classifier():
    return load_classifier()


def play_text(text, key):
    saved_key = f"audio_{key}"
    if st.button("Speak this text", key=key, disabled=not bool(text)):
        try:
            audio, mime = synthesize(text)
            st.session_state[saved_key] = (text, audio, mime)
        except Exception as exc:
            st.warning(f"Audio unavailable: {exc}. The text remains visible.")
    saved = st.session_state.get(saved_key)
    if saved and saved[0] == text:
        st.audio(saved[1], format=saved[2])


def show_recognition(result):
    if result.status == "high":
        st.success(result.phrase)
        st.caption(f"Predicted gesture: {result.label} · model score: {result.confidence:.0%}")
        play_text(result.phrase, f"speak_{result.label}")
    elif result.status == "confirm":
        st.warning(f"Possible gesture: {result.label} ({result.confidence:.0%} model score). Confirm before speaking.")
        if st.checkbox("I confirm this is the intended message", key="confirm_gesture"):
            st.info(result.phrase)
            play_text(result.phrase, "speak_confirmed")
    else:
        st.info("Could not confidently recognize the gesture. Try again with one hand and good lighting.")


class LiveEngine:
    """The video callback owns the landmarker; the UI reads a small thread-safe result."""

    def __init__(self, model, phrases):
        self.model = model
        self.phrases = phrases
        self.lock = threading.Lock()
        self.extractor = None
        self.votes = deque(maxlen=5)
        self.latest = None
        self.message = "Waiting for one hand in the camera frame."

    def process(self, frame):
        import av
        import cv2

        bgr = frame.to_ndarray(format="bgr24")
        try:
            if self.extractor is None:
                self.extractor = HandExtractor()
            features, state, points = self.extractor.extract(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if features is None:
                self.votes.clear()
                with self.lock:
                    self.latest = None
                    self.message = "Move one hand into view." if state == "no_hand" else "Show only one hand."
            else:
                result = predict(features, self.model, self.phrases)
                self.votes.append(result.label if result.status == "high" else None)
                stable = len(self.votes) >= 4 and list(self.votes)[-4:] == [result.label] * 4
                with self.lock:
                    self.latest = result if stable and result.status == "high" else None
                    self.message = "Gesture held steady." if self.latest else "Hold a supported gesture steady for a moment."
                height, width = bgr.shape[:2]
                for x, y in points:
                    cv2.circle(bgr, (int(x * width), int(y * height)), 3, (62, 220, 112), -1)
        except Exception as exc:
            with self.lock:
                self.latest = None
                self.message = f"Camera processing unavailable: {exc}"
        return av.VideoFrame.from_ndarray(bgr, format="bgr24")

    def snapshot(self):
        with self.lock:
            return self.latest, self.message


def sign_mode(phrases):
    st.header("Sign or gesture → text and speech")
    st.write("This model recognizes only team-collected, verified labels. It does not translate unrestricted sign language.")
    capture, live, fallback = st.tabs(["Take a picture", "Live camera", "Demo backup"])
    with fallback:
        st.info("Manual demo backup. This selection is not an AI prediction.")
        label = st.selectbox("Choose a prepared phrase", list(phrases))
        st.write(phrases[label])
        play_text(phrases[label], "speak_fallback")
    if not LANDMARK_MODEL.is_file() or not CLASSIFIER.is_file():
        with capture:
            st.warning("Gesture recognition needs the hand model and your trained gesture classifier. Follow the setup steps in README.md.")
        with live:
            st.warning("Train the gesture classifier before starting live recognition.")
        return
    try:
        model = classifier()
    except Exception as exc:
        st.error(f"Could not load gesture classifier: {exc}")
        return
    with capture:
        photo = st.camera_input("Show one supported gesture and take a picture")
        if photo and st.button("Recognize picture"):
            from PIL import Image

            extractor = None
            try:
                extractor = HandExtractor()
                features, state, _ = extractor.extract(np.array(Image.open(photo).convert("RGB")))
                if features is None:
                    st.info("Move one hand into view." if state == "no_hand" else "Show only one hand.")
                else:
                    st.session_state["photo_result"] = predict(features, model, phrases)
            except Exception as exc:
                st.error(f"Recognition failed: {exc}")
            finally:
                if extractor:
                    extractor.close()
        if result := st.session_state.get("photo_result"):
            show_recognition(result)
    with live:
        st.caption("Live video requires browser camera permission and a secure connection outside localhost.")
        try:
            from streamlit_webrtc import webrtc_streamer, WebRtcMode

            engine = LiveEngine(model, phrases)
            webrtc_streamer(key="gesture-live", mode=WebRtcMode.SENDRECV,
                           video_frame_callback=engine.process,
                           media_stream_constraints={"video": True, "audio": False},
                           async_processing=True)

            @st.fragment(run_every="0.5s")
            def live_result():
                result, message = engine.snapshot()
                if result:
                    st.success(result.phrase)
                    st.caption(f"Predicted {result.label} · model score {result.confidence:.0%}")
                    play_text(result.phrase, "speak_live")
                else:
                    st.info(message)

            live_result()
        except ImportError:
            st.warning("Install the optional live video package from requirements.txt. Picture mode remains available.")


def audio_input(label, key):
    clip = st.audio_input(label, key=key)
    if clip and st.button("Transcribe recording", key=f"transcribe_{key}"):
        try:
            with st.spinner("Transcribing locally (the first model download may take time)…"):
                st.session_state[f"text_{key}"] = transcribe(clip.getvalue(), language="en")
        except Exception as exc:
            st.error(f"Transcription failed: {exc}. Record and try again.")
    return st.session_state.get(f"text_{key}", "")


def speech_mode():
    st.header("Speech → text")
    text = audio_input("Record a short reply", "speech")
    if text:
        st.text_area("Transcript", value=text, height=130)
        play_text(text, "speak_transcript")


def voice_mode():
    st.header("Voice-first interaction")
    st.write("Record a short message. The app reads back a simple, scripted acknowledgement or repeats what it heard.")
    text = audio_input("Record your message", "voice")
    if text:
        st.text_area("What I heard", value=text, height=110)
        reply = select_response(text)
        st.write("Spoken response text:", reply)
        if st.session_state.get("voice_response_text") != reply:
            try:
                audio, mime = synthesize(reply)
                st.session_state["voice_response_audio"] = (audio, mime)
                st.session_state["voice_new_audio"] = True
            except Exception as exc:
                st.session_state.pop("voice_response_audio", None)
                st.warning(f"Audio unavailable: {exc}. The response remains readable.")
            st.session_state["voice_response_text"] = reply
        if recorded := st.session_state.get("voice_response_audio"):
            st.audio(recorded[0], format=recorded[1], autoplay=st.session_state.pop("voice_new_audio", False))


mode = st.radio("Communication mode", ["Sign / gesture", "Speech to text", "Voice-first"], horizontal=True)
try:
    phrases = load_phrases()
    if mode == "Sign / gesture":
        sign_mode(phrases)
    elif mode == "Speech to text":
        speech_mode()
    else:
        voice_mode()
except Exception as exc:
    st.error(f"Could not start this mode: {exc}")

st.divider()
st.caption("Privacy: live camera frames and microphone recordings are not saved by this app. Data collection is a separate explicit command.")
