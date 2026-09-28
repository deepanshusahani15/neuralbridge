# Neuralbridge

Neuralbridge is a hackathon MVP for short two-way exchanges across communication modes. It recognizes a **limited, team-trained gesture vocabulary**, displays a hearing person's spoken reply as text, and provides a simple voice-first readback. It is not a general Indian Sign Language translator or an emergency communication system.

## What runs today

- Streamlit UI with picture capture, a live camera option, speech-to-text, voice-first response, and a clearly labeled manual demo backup.
- MediaPipe Hand Landmarker features plus a separately trained Random Forest classifier. No fabricated model or sample dataset is bundled.
- Local faster-whisper speech transcription. Model weights download on first use; later inference can run offline.
- Local TTS through macOS `say` and `afconvert` or Linux `espeak-ng`; readable text remains if audio fails.
- One-hand handling, confidence gates, and four consistent high-score frames before live results appear.

## Setup on macOS or Linux

Use Python 3.11 or 3.12 on a laptop with a camera and microphone. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/download_hand_model.py
streamlit run app.py
```

On Linux, install `espeak-ng` through your system package manager for speech output. On macOS, `say` and `afconvert` are normally built in. Run on `localhost` for camera and microphone access; remote use requires HTTPS. The first speech transcription downloads a small model from its model host, so prepare that while connected to the internet before demo day.

The app can launch without a trained gesture classifier. Its gesture AI modes clearly say that training is required; speech and the manual backup still work.

## Prepare gesture data

1. Agree on visible, static gestures using appropriate Indian Sign Language references and people familiar with the language. The keys in `data/phrase_map.json` are **proposed product labels**, not a claim that a particular handshape is the correct sign. Remove labels that cannot be verified or reliably separated using one static hand. Some concepts need movement or both hands and cannot be captured by this baseline.
2. Edit `data/phrase_map.json` to the agreed labels and phrases. Keep at least two labels. Capture each label from at least three participants, ideally 80–150 samples per label, with different lighting and distances. Ask volunteers for consent. Do not enter personal names as participant IDs.
3. On the **local laptop** (the collector uses its camera directly), run a command per label, participant, and session:

```bash
python scripts/collect.py --label HELLO --participant volunteer_a --count 100
python scripts/collect.py --label HELP --participant volunteer_a --count 100
# Repeat every selected class with volunteer_b and volunteer_c.
python scripts/train.py
```

The collector saves only a CSV of landmarks and metadata in `data/processed/landmarks.csv`. `q` ends capture early. Avoid holding a completely static pose for every sample: vary orientation and distance, then check labels and remove mistakes. Training rejects datasets that cannot hold out whole participants (or, with fewer people, whole recording sessions) while retaining all classes. Its output is `models/sign_classifier.joblib` and `models/evaluation.json`. These generated files are ignored by Git, so preserve and share the known-good trained model with the team through an approved private route. Only load a model file produced by your team because joblib files use Python pickle.

Read `models/evaluation.json` for held-out accuracy, per-class precision and recall, and the confusion matrix. A session split with the same participants is weaker evidence than a participant split. Class probabilities are uncalibrated and can still be confident on an unknown gesture; test strangers and out-of-vocabulary signs before claiming reliability.

## Three-minute demo

1. Start the known-good build before judges join. Choose **Sign / gesture → Take a picture**. Show a supported gesture and press **Recognize picture**. If using **Live camera**, hold a supported gesture for a moment.
2. Show the phrase and model score. Speak a high-confidence result, or confirm a medium-confidence picture manually.
3. Switch to **Speech to text**, record the other person's spoken reply, then press **Transcribe recording**. Show the visible transcript.
4. Switch to **Voice-first**, record a short message, transcribe it, and press **Speak this text** to read the selected reply aloud.
5. If the camera or model fails, use **Demo backup** and explicitly tell judges it is a manual selection, then continue with the speech flow. Never present the backup as recognition.

## Project layout

| Path | Purpose |
| --- | --- |
| `app.py` | Three Streamlit modes and live camera display |
| `backend/sign_recognition.py` | Hand landmarks, normalization, classifier inference |
| `backend/speech_to_text.py` | Short local audio transcription |
| `backend/text_to_speech.py` | Local speech output |
| `backend/phrase_mapper.py`, `validation.py`, `voice_response.py` | Phrases, thresholds, scripted acknowledgement |
| `scripts/download_hand_model.py` | Fetch the official MediaPipe task asset |
| `scripts/collect.py`, `scripts/train.py` | Explicit labeled collection and group-held-out training |
| `data/phrase_map.json` | Editable vocabulary and spoken phrases |
| `tests/` | Core behavior checks |

Run the core checks with `python -m unittest discover -s tests -v`.

## Privacy and limitations

Live app frames and microphone audio are processed in memory or a temporary file, and not stored. Data collection is a separate explicit command; the resulting CSV should be treated as participant data even though it contains landmarks rather than images. Webcam access requires permission. This version recognizes one visible static hand; signs involving movement, facial expression, two hands, or sentence grammar are out of scope. Poor lighting, occlusion, untested users, noise, accents, and OS TTS availability affect results. A model score is not a calibrated confidence guarantee.

## Technical references

- [MediaPipe Hand Landmarker Python guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)
- [Streamlit camera input](https://docs.streamlit.io/develop/api-reference/widgets/st.camera_input) and [audio input](https://docs.streamlit.io/develop/api-reference/widgets/st.audio_input)
- [streamlit-webrtc video callback](https://github.com/whitphx/streamlit-webrtc)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [ISLRTC](https://islrtc.nic.in/faq/) for starting the sign-reference verification process
