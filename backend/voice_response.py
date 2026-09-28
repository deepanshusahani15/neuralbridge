"""Transparent scripted responses for the audio-first demonstration."""


def select_response(transcript):
    words = set(transcript.lower().strip(" .!?\n").split())
    if not words:
        return "I did not hear a message. Please try again."
    if "help" in words:
        return "I heard that you need help. Please tell the person nearby what you need."
    if "water" in words:
        return "I heard that you need water."
    if "hello" in words or "hi" in words:
        return "Hello. Your voice message was received."
    return f"I heard: {transcript}"
