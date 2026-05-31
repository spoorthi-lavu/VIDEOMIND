# VideoMind

> [LinkedIn Post](https://www.linkedin.com/feed/update/urn:li:activity:7464880551888404480/)

An end-to-end multimodal AI platform for video forensics, deepfake detection,
liveness analysis, emotion recognition, and speech transcription — with a
built-in video chatbot, live webcam mode, 7-language support, and Braille
accessibility.

## Modes

| Mode | Description |
|------|-------------|
| `video_to_text` | Whisper transcription + CLIP captions + FAISS chatbot |
| `forensic` | Full pipeline — all modules, final authenticity verdict |
| `live` | Real-time webcam deepfake + rPPG + emotion analysis |

## Features

- **Deepfake Detection** — pretrained HuggingFace deepfake detection model with CNN/FFT fusion. Custom training on FaceForensics++ and CelebDF was attempted but switched to pretrained due to class imbalance.
- **Emotion Recognition** — frame-level emotion timeline with dominant emotion tracking.
- **Speech Transcription** — OpenAI Whisper with timestamped segments.
- **Video Chatbot** — FAISS-powered semantic search over video content.
- **Scene Detection** — CLIP-based semantic scene boundary detection.
- **Sparse MoE Fusion** — Context-Aware Temporal Mixture of Experts for final verdict (work in progress).
- **DRE / UNet Reconstruction** — out-of-distribution scoring (work in progress).
- **Live Webcam Mode** — real-time frame-by-frame analysis.
- **7-Language Support** — Hindi, French, German, Spanish, Korean, Japanese.
- **Braille Accessibility** — full transcript output in Braille.

## Tech Stack

- **Backend:** Flask, PyTorch, HuggingFace Transformers
- **Vision:** CLIP, BLIP, MediaPipe, OpenCV
- **Audio:** Whisper, Librosa
- **Search:** FAISS, SentenceTransformers
- **Deployment:** JupyterHub + ngrok

## Presented At

AI for Next Gen Conclave — CR Rao AIMSCS & JNTUH

## Known Limitations

- Emotion classifier has a slight bias toward the angry class.
- Deepfake module uses a pretrained model; custom training was attempted but deprioritised.
- MoE fusion and DRE reconstruction are works in progress.
- rPPG requires sufficient face frames for reliable pulse estimation.

## Setup

```bash
pip install -r requirements.txt
python app.py
```
