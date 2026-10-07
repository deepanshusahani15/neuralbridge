"""Neuralbridge three-mode communication MVP."""

import numpy as np
import streamlit as st

from backend.live_camera import LiveEngine
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


def live_camera(model, phrases):
    st.caption("Press START and allow camera access. Press STOP to release the camera.")
    if model is None:
        st.info("Camera preview is available. Gesture recognition requires the hand model and a trained classifier; see README.md.")
    try:
        from streamlit_webrtc import WebRtcMode, webrtc_streamer
    except ImportError:
        st.error("Live camera dependencies are missing. Run: python -m pip install -r requirements.txt")
        return

    context = webrtc_streamer(
        key="gesture-live",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: LiveEngine(model, phrases),
        media_stream_constraints={
            "video": {"width": {"ideal": 640}, "height": {"ideal": 480},
                      "frameRate": {"ideal": 15, "max": 20}},
            "audio": False,
        },
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
    )

    @st.fragment(run_every="0.5s")
    def live_result():
        if not context.state.playing:
            st.info("Camera stopped. Press START to begin.")
            return
        engine = context.video_processor
        if engine is None or model is None:
            st.info("Camera preview only." if model is None else "Connecting camera…")
            return
        result, message = engine.snapshot()
        if result:
            st.success(result.phrase)
            st.caption(f"Predicted {result.label} · model score {result.confidence:.0%}")
            play_text(result.phrase, "speak_live")
        else:
            st.info(message)

    live_result()
    st.caption("If video does not connect, allow camera access in your browser and close other apps using it. Remote access requires HTTPS; some networks also require a TURN server.")


def sign_mode(phrases):
    st.header("Sign or gesture → text and speech")
    st.write("This model recognizes only team-collected, verified labels. It does not translate unrestricted sign language.")
    live, capture, fallback = st.tabs(["Live camera", "Take a picture", "Demo backup"])
    with fallback:
        st.info("Manual demo backup. This selection is not an AI prediction.")
        label = st.selectbox("Choose a prepared phrase", list(phrases))
        st.write(phrases[label])
        play_text(phrases[label], "speak_fallback")
    model = None
    if LANDMARK_MODEL.is_file() and CLASSIFIER.is_file():
        try:
            model = classifier()
        except Exception as exc:
            st.warning(f"Could not load gesture classifier: {exc}")
    with live:
        live_camera(model, phrases)
    if model is None:
        with capture:
            st.warning("Picture recognition needs the hand model and your trained gesture classifier. Follow README.md.")
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
