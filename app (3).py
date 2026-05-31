import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"  # skip stuck GPU 0

"""
VideoMind — Mode-Aware Pipeline
================================
Three modes:
  1. video_to_text   — Whisper + CLIP captions + unified representation only
  2. forensic        — Full pipeline (all GPUs, all modules)
  3. live            — Webcam: CLIP + DeepfakeCNN + rPPG + Emotion (sliding window)

GPU Layout (4× RTX 2080 Ti, 11 GB each):
  GPU 0 (cuda:0): CLIP only
  GPU 1 (cuda:1): Whisper large + Qwen2-VL-2B FP16
  GPU 2 (cuda:2): SparseMoE + DRE + Embeddings + DeepfakeCNN + TemporalLSTM
  GPU 3 (cuda:3): EmotionClassifier + overflow buffer

Routes:
  POST /upload_file          — file upload, param: mode
  POST /upload               — YouTube URL, param: mode
  GET  /status/<jid>
  GET  /results/text/<jid>
  GET  /results/forensic/<jid>
  GET  /results/live/<sid>   — live session summary
  POST /webcam/start         — start live session → {session_id}
  POST /webcam/frame         — accept base64 frame → rolling scores
  POST /webcam/stop/<sid>    — end session → summary
  GET  /forensic/<jid>       — raw forensic JSON
  GET  /analysis/*/<jid>     — individual module JSON
  POST /chat/<video_hash>
  GET  /translate/<lang>
  etc.
"""

import os, sys, uuid, re, threading, traceback, json, time, base64
from pathlib import Path
from typing  import Optional, Dict, List, Tuple, Any
from collections import Counter
from io import BytesIO

# ── Work-drive paths ───────────────────────────────────────
WORK         = "/home/jovyan/work"
PKG_PATH     = f"{WORK}/python_user/lib/python3.11/site-packages"
if PKG_PATH not in sys.path:
    sys.path.insert(0, PKG_PATH)

os.environ["HF_HOME"]            = f"{WORK}/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = f"{WORK}/hf_cache"
os.environ["TORCH_HOME"]         = f"{WORK}/torch_cache"

UPLOAD_FOLDER = f"{WORK}/videomind/uploads"
CACHE_DIR     = f"{WORK}/videomind/feature_cache"
INDEX_DIR     = f"{WORK}/videomind/video_indexes"
MODELS_DIR    = f"{WORK}/models"

for _d in [UPLOAD_FOLDER, CACHE_DIR, INDEX_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Core imports ───────────────────────────────────────────
from flask import (Flask, render_template, request, send_from_directory,
                   jsonify, session, Response)
from werkzeug.utils import secure_filename

import numpy as np
from PIL import Image
from moviepy import VideoFileClip

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from sklearn.metrics.pairwise import cosine_similarity

import whisper
from sentence_transformers import SentenceTransformer
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    CLIPProcessor, CLIPModel,
)
import yt_dlp
from deep_translator import GoogleTranslator

# ── Optional deps ──────────────────────────────────────────
try:    import pyarrow as pa; import pyarrow.parquet as pq; HAS_ARROW = True
except: HAS_ARROW = False
try:    import h5py;    HAS_HDF5    = True
except: HAS_HDF5    = False
try:    import librosa; HAS_LIBROSA = True
except: HAS_LIBROSA = False
try:    import cv2;     HAS_CV2     = True
except: HAS_CV2     = False
try:    import faiss;   HAS_FAISS   = True
except: HAS_FAISS   = False
try:    import scipy.fftpack as fftpack; HAS_SCIPY = True
except: HAS_SCIPY = False

# ─────────────────────────────────────────────────────────
#  FLASK
# ─────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"mp4","mov","mkv","avi","webm"}
VALID_MODES = {"video_to_text", "forensic", "live"}

def allowed_file(f):
    return "." in f and f.rsplit(".",1)[1].lower() in ALLOWED_EXTENSIONS

# ─────────────────────────────────────────────────────────
#  GPU TOPOLOGY
# ─────────────────────────────────────────────────────────
NUM_GPUS = torch.cuda.device_count()
HAS_CUDA = torch.cuda.is_available()

def _free_mib(i: int) -> float:
    if not HAS_CUDA or i >= NUM_GPUS: return 0.0
    p = torch.cuda.get_device_properties(i)
    return (p.total_memory - torch.cuda.memory_reserved(i)) / 1024**2

_free = [_free_mib(i) for i in range(NUM_GPUS)]
print(f"[VideoMind] Free VRAM (MiB): { {i: round(_free[i]) for i in range(NUM_GPUS)} }")

GPU_CLIP         = "cuda:0"
GPU_WHISPER      = "cuda:0"
GPU_QWEN         = "cuda:0"
GPU_FUSION       = "cuda:0"
GPU_EMOTION      = "cuda:0"
GPU_CLIP_WHISPER = "cuda:0"

FRAME_STEP_SECONDS = 2.0
MAX_FRAMES         = 150
KEYFRAME_INTERVAL  = 2
VLM_MAX_TOKENS     = 100
MAX_DEEPFAKE_KF    = 20

# ─────────────────────────────────────────────────────────
#  JOB STORE + LIVE SESSION STORE
# ─────────────────────────────────────────────────────────
jobs: Dict[str, dict] = {}
jobs_lock = threading.Lock()

# Live sessions: session_id → rolling state
live_sessions: Dict[str, dict] = {}
live_sessions_lock = threading.Lock()

def set_job(jid: str, **kw):
    with jobs_lock:
        jobs[jid].update(kw)

# ═════════════════════════════════════════════════════════
#  FEATURE STORE
# ═════════════════════════════════════════════════════════
class FeatureStore:
    def __init__(self, d=CACHE_DIR):
        self.cache_dir = Path(d); self.cache_dir.mkdir(exist_ok=True)
        self._mem: Dict[str, np.ndarray] = {}
        self.backend = "arrow" if HAS_ARROW else ("hdf5" if HAS_HDF5 else "memory")

    def _p(self, k):
        return self.cache_dir / f"{re.sub(r'[^a-zA-Z0-9_-]','_',k)}.parquet"

    def get(self, k):
        if self.backend == "arrow":
            p = self._p(k)
            if p.exists():
                return np.array(pq.read_table(p)["embedding"].to_pylist()[0])
        elif self.backend == "hdf5":
            h = self.cache_dir / "store.h5"
            if h.exists():
                with h5py.File(h,"r") as f:
                    if k in f: return f[k][:]
        return self._mem.get(k)

    def put(self, k, v):
        self._mem[k] = v
        if self.backend == "arrow":
            pq.write_table(pa.table({"key":[k],"embedding":[v.tolist()]}), self._p(k))
        elif self.backend == "hdf5":
            with h5py.File(self.cache_dir/"store.h5","a") as f:
                f.require_dataset(k, data=v, shape=v.shape, dtype=v.dtype)

feature_store = FeatureStore()

# ═════════════════════════════════════════════════════════
#  MODEL REGISTRY
# ═════════════════════════════════════════════════════════
class ModelRegistry:
    def __init__(self):
        self._m: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def get_or_load(self, key, loader_fn):
        with self._lock:
            if key not in self._m:
                print(f"[Registry] Loading {key} ...")
                self._m[key] = loader_fn()
                print(f"[Registry] ✓ {key}")
            return self._m[key]

    def unload(self, key):
        with self._lock:
            if key in self._m:
                del self._m[key]
                torch.cuda.empty_cache()

_registry = ModelRegistry()

# ─────────────────────────────────────────────────────────
#  MODEL LOADERS
# ─────────────────────────────────────────────────────────

def _load_clip(device):
    path = f"{MODELS_DIR}/clip-vit-base-patch32"
    src  = path if Path(path+"/config.json").exists() else "openai/clip-vit-base-patch32"
    return CLIPProcessor.from_pretrained(src), CLIPModel.from_pretrained(src).to(device).eval()

def _load_qwen2vl_2b(device):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    src   = "Qwen/Qwen2-VL-2B-Instruct"
    proc  = AutoProcessor.from_pretrained(src, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        src, torch_dtype=torch.float16, device_map={"": device}, trust_remote_code=True).eval()
    return proc, model

def _load_blip(device):
    path = f"{MODELS_DIR}/blip-image-captioning-base"
    src  = path if Path(path+"/config.json").exists() else "Salesforce/blip-image-captioning-base"
    return BlipProcessor.from_pretrained(src), BlipForConditionalGeneration.from_pretrained(src).to(device).eval()

def _load_whisper_large(device):
    root = f"{WORK}/hf_cache/whisper"
    if not Path(root).exists(): root = "/home/jovyan/.cache/whisper"
    return whisper.load_model("small", device=device, download_root=root)

def _load_embed(device):
    path = f"{MODELS_DIR}/all-MiniLM-L6-v2"
    src  = path if Path(path+"/config.json").exists() else "all-MiniLM-L6-v2"
    return SentenceTransformer(src, device=device)

def get_clip():
    return _registry.get_or_load(f"clip_{GPU_CLIP}", lambda: _load_clip(GPU_CLIP))

def get_qwen():
    # VRAM fix: skip Qwen2-VL entirely, use BLIP to free ~8GB for Whisper
    print("[VLM] Using BLIP (Qwen2-VL disabled to save VRAM)")
    return _registry.get_or_load(f"blip_{GPU_QWEN}", lambda: _load_blip(GPU_QWEN))

def get_whisper():
    return _registry.get_or_load(f"whisper_{GPU_WHISPER}", lambda: _load_whisper_large(GPU_WHISPER))

def get_embed():
    return _registry.get_or_load(f"embed_{GPU_FUSION}", lambda: _load_embed(GPU_FUSION))

# ═════════════════════════════════════════════════════════
#  MODULE 1: DEEPFAKE DETECTION CNN + FFT
# ═════════════════════════════════════════════════════════

class DeepfakeDetectorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights=None)
        self.features = backbone.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 256), nn.Hardswish(), nn.Dropout(0.3),
            nn.Linear(256, 64),  nn.Hardswish(),
            nn.Linear(64, 2),
        )
        self.freq_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(256, 1), nn.Sigmoid(),
        )
        self.fft_branch = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(8), nn.Flatten(),
            nn.Linear(512, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )
        self.fusion_weights = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]))

    def _fft_features(self, x):
        gray    = x.mean(dim=1, keepdim=True)
        fft     = torch.fft.fft2(gray)
        mag     = torch.abs(fft) + 1e-8
        log_mag = torch.log(mag)
        log_mag = torch.fft.fftshift(log_mag, dim=(-2,-1))
        mn = log_mag.flatten(2).min(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
        mx = log_mag.flatten(2).max(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
        return (log_mag - mn) / (mx - mn + 1e-8)

    def forward(self, x):
        feats     = self.features(x)
        pooled    = self.pool(feats)
        logits    = self.classifier(pooled)
        probs     = F.softmax(logits, dim=-1)
        spatial_p = probs[:, 1]
        freq_score = self.freq_conv(x).squeeze(-1)
        fft_feats  = self._fft_features(x)
        fft_resized= F.interpolate(fft_feats, size=(32,32), mode='bilinear', align_corners=False)
        fft_score  = self.fft_branch(fft_resized).squeeze(-1)
        w = F.softmax(self.fusion_weights, dim=0)
        fake_prob = w[0]*spatial_p + w[1]*freq_score + w[2]*fft_score
        return fake_prob, probs

_hf_deepfake = None

def _get_deepfake_detector():
    global _hf_deepfake
    if _hf_deepfake is None:
        model = DeepfakeDetectorCNN()
        ckpt = Path(f"{WORK}/videomind/deepfake_detector.pth")
        if ckpt.exists():
            state_dict = torch.load(ckpt, map_location=GPU_FUSION, weights_only=False)
            missing, unexpected = model.load_state_dict(state_dict, strict=True)
            print(f"[Deepfake] ✓ Loaded correctly — missing={len(missing)} unexpected={len(unexpected)}")
        else:
            print("[Deepfake] No checkpoint found - random init")
        _hf_deepfake = model.to(GPU_FUSION).eval()
    return _hf_deepfake

# ── MediaPipe Face Detection ───────────────────────────────
_mp_face = None

def _get_mp_face():
    global _mp_face
    if _mp_face is None:
        try:
            import mediapipe as mp
            _mp_face = mp.solutions.face_detection.FaceDetection(
                model_selection=1,        # 1 = full range model (better for varied angles)
                min_detection_confidence=0.3)  # low threshold = catches more faces
            print("[FaceNet] ✓ MediaPipe face detector loaded")
        except Exception as e:
            print(f"[FaceNet] MediaPipe load failed: {e}")
    return _mp_face


_haar_cascade = None
_face_net = None

def _get_haar():
    global _haar_cascade
    if _haar_cascade is None and HAS_CV2:
        cc_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _haar_cascade = cv2.CascadeClassifier(cc_path)
    return _haar_cascade

if HAS_CV2:
    try:
        import urllib.request
        proto     = f"{WORK}/models/deploy.prototxt"
        caffemodel= f"{WORK}/models/res10_300x300_ssd_iter_140000.caffemodel"
        Path(f"{WORK}/models").mkdir(exist_ok=True)
        if not Path(proto).exists():
            urllib.request.urlretrieve("https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt", proto)
        if not Path(caffemodel).exists():
            urllib.request.urlretrieve("https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel", caffemodel)
        _face_net = cv2.dnn.readNetFromCaffe(proto, caffemodel)
        print("[FaceNet] ✓ DNN face detector loaded")
    except Exception as e:
        print(f"[FaceNet] ✗ {e} — Haar fallback only")
        _face_net = None

def extract_faces(img: Image.Image) -> List[Image.Image]:
    if not HAS_CV2: return []
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    faces = []
    try:
        blob = cv2.dnn.blobFromImage(arr, 1.0, (300,300), (104.0,177.0,123.0), swapRB=False)
        _face_net.setInput(blob)
        dets = _face_net.forward()
        for i in range(dets.shape[2]):
            conf = float(dets[0,0,i,2])
            if conf < 0.5: continue
            x0=max(0,int(dets[0,0,i,3]*w)); y0=max(0,int(dets[0,0,i,4]*h))
            x1=min(w,int(dets[0,0,i,5]*w)); y1=min(h,int(dets[0,0,i,6]*h))
            if x1-x0>20 and y1-y0>20:
                pad=int(0.15*min(x1-x0,y1-y0))
                faces.append(Image.fromarray(arr[max(0,y0-pad):min(h,y1+pad),max(0,x0-pad):min(w,x1+pad)]))
        if faces:
            del arr; return faces[:4]
    except Exception:
        pass
    cascade=_get_haar()
    if cascade is None or cascade.empty():
        del arr; return []
    gray=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
    rects=cascade.detectMultiScale(gray,scaleFactor=1.05,minNeighbors=3,minSize=(30,30))
    del gray
    if len(rects)>0:
        for (x,y,w2,h2) in rects[:4]:
            pad=int(0.15*min(w2,h2))
            faces.append(Image.fromarray(arr[max(0,y-pad):min(h,y+h2+pad),max(0,x-pad):min(w,x+w2+pad)]))
    del arr
    return faces[:4]
# ── HuggingFace deepfake pipeline ────────────────────────
from transformers import pipeline as hf_pipeline
_deepfake_pipe = None

def _get_deepfake_pipe():
    global _deepfake_pipe
    if _deepfake_pipe is None:
        print("[Deepfake] Loading dima806/deepfake_vs_real_image_detection on CPU...")
        _deepfake_pipe = hf_pipeline(
            "image-classification",
            model="dima806/deepfake_vs_real_image_detection",
            device=-1
        )
        print("[Deepfake] ✓ Pipeline ready")
    return _deepfake_pipe

def detect_deepfake_frame(img: Image.Image) -> dict:
    """
    Accepts PIL Image. Uses dima806 HF pipeline on CPU.
    Returns dict compatible with forensic + live pipelines.
    """
    faces   = extract_faces(img)
    targets = faces if faces else [img]
    mode    = "face" if faces else "full_frame"
    pipe    = _get_deepfake_pipe()

    scores_fake = []
    for patch in targets[:4]:
        preds = pipe(patch)
        for p in preds:
            if "FAKE" in p["label"].upper() or "DEEPFAKE" in p["label"].upper():
                scores_fake.append(p["score"])
                break
        else:
            scores_fake.append(0.0)

    fake_score = float(np.mean(scores_fake)) if scores_fake else 0.0

    return {
        "fake_score":      round(fake_score, 4),
        "verdict":         "FAKE" if fake_score > 0.5 else "REAL",
        "face_count":      len(faces),
        "mode":            mode,
        "frame_score":     round(scores_fake[0], 4) if scores_fake else 0.0,
        "per_face_scores": [round(s, 4) for s in scores_fake],
    }

# ═════════════════════════════════════════════════════════
#  MODULE 2: TEMPORAL LSTM
# ═════════════════════════════════════════════════════════

class TemporalLSTMScorer(nn.Module):
    def __init__(self, feat_dim=512, hidden=128):
        super().__init__()
        self.proj  = nn.Linear(feat_dim, hidden)
        self.lstm  = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True,
                             dropout=0.1, bidirectional=True)
        self.score = nn.Sequential(
            nn.Linear(hidden*2, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, seq):
        x, _ = self.lstm(self.proj(seq))
        return self.score(x).squeeze(-1)

_temporal_lstm = None

def _get_temporal_lstm():
    global _temporal_lstm
    if _temporal_lstm is None:
        _temporal_lstm = TemporalLSTMScorer(feat_dim=512, hidden=128).to(GPU_FUSION).eval()
        ckpt = Path(f"{WORK}/videomind/temporal_lstm.pth")
        if ckpt.exists():
            _temporal_lstm.load_state_dict(torch.load(ckpt, map_location=GPU_FUSION))
    return _temporal_lstm

def _find_anomaly_segments(scores, timestamps, threshold, min_gap_sec=2.0):
    if len(scores) == 0: return []
    in_window = False; window_start = 0.0; window_scores = []; segments = []
    for i, (t, sc) in enumerate(zip(timestamps, scores)):
        if sc > threshold:
            if not in_window:
                in_window = True; window_start = t; window_scores = []
            window_scores.append(sc)
        else:
            if in_window:
                seg_end = timestamps[i-1] if i > 0 else t
                if segments and (window_start - segments[-1]["end"]) < min_gap_sec:
                    segments[-1]["end"]        = seg_end
                    segments[-1]["duration"]   = round(seg_end - segments[-1]["start"], 2)
                    segments[-1]["peak_score"] = round(max(segments[-1]["peak_score"], float(np.max(window_scores))), 4)
                else:
                    segments.append({"start": round(window_start,2), "end": round(seg_end,2),
                                     "duration": round(seg_end-window_start,2),
                                     "peak_score": round(float(np.max(window_scores)),4),
                                     "mean_score": round(float(np.mean(window_scores)),4),
                                     "reason": "Temporally localized LSTM anomaly window"})
                in_window = False
    if in_window and window_scores:
        segments.append({"start": round(window_start,2), "end": round(timestamps[-1],2),
                         "duration": round(timestamps[-1]-window_start,2),
                         "peak_score": round(float(np.max(window_scores)),4),
                         "mean_score": round(float(np.mean(window_scores)),4),
                         "reason": "Temporally localized LSTM anomaly window"})
    return segments

def detect_temporal_inconsistencies(clip_embs, timestamps):
    if len(clip_embs) < 3:
        return {"anomaly_scores":[], "anomaly_segments":[], "suspicious_jumps":[],
                "max_anomaly":0.0, "mean_anomaly":0.0, "n_suspicious":0}
    emb_matrix = np.stack(clip_embs)
    deltas = [float(1.0 - cosine_similarity(emb_matrix[i-1:i], emb_matrix[i:i+1])[0,0])
              for i in range(1, len(emb_matrix))]
    try:
        with torch.no_grad():
            lstm = _get_temporal_lstm()
            D    = emb_matrix.shape[1]
            padded = emb_matrix[:, :512] if D >= 512 else np.pad(emb_matrix, ((0,0),(0,512-D)))
            seq  = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(GPU_FUSION)
            lstm_scores = lstm(seq)[0].cpu().numpy()
        torch.cuda.empty_cache()
    except Exception:
        d_arr = np.array([0.0] + deltas)
        lstm_scores = d_arr / (d_arr.max() + 1e-9)
    threshold = float(np.mean(lstm_scores) + 1.5 * np.std(lstm_scores))
    anomaly_segments = _find_anomaly_segments(lstm_scores, timestamps, threshold)
    anomaly_scores   = [{"t": round(t,2), "score": round(float(sc),4)}
                        for t, sc in zip(timestamps, lstm_scores)]
    suspicious = [{"t": round(t,2), "anomaly_score": round(float(sc),4),
                   "frame_delta": round(deltas[i-1] if i>0 else 0.0, 4),
                   "reason": "High temporal discontinuity"}
                  for i, (t, sc) in enumerate(zip(timestamps, lstm_scores)) if float(sc) > threshold]
    return {"anomaly_scores": anomaly_scores, "anomaly_segments": anomaly_segments,
            "suspicious_jumps": suspicious, "max_anomaly": round(float(lstm_scores.max()),4),
            "mean_anomaly": round(float(lstm_scores.mean()),4), "n_suspicious": len(suspicious),
            "n_anomaly_segments": len(anomaly_segments), "threshold_used": round(threshold,4)}

# ═════════════════════════════════════════════════════════
#  MODULE 3: SCENE DETECTION
# ═════════════════════════════════════════════════════════

def detect_scenes(frames_ts, clip_embs, pixel_threshold=0.35, embed_threshold=0.25):
    scenes=[]; boundaries=[]; scene_start=frames_ts[0][0] if frames_ts else 0.0
    scene_idx=0; prev_arr=None; prev_gray=None
    for i, (t, img) in enumerate(frames_ts):
        arr  = np.array(img.resize((128,128))).astype(np.float32)/255.0
        gray = np.mean(arr, axis=2)
        if prev_arr is None:
            prev_arr=arr; prev_gray=gray; continue
        pixel_diff = float(np.mean(np.abs(arr-prev_arr)))
        embed_diff = 0.0
        if i < len(clip_embs) and i-1 < len(clip_embs):
            sim = cosine_similarity(clip_embs[i-1:i], clip_embs[i:i+1])[0,0]
            embed_diff = float(1.0-sim)
        is_boundary = (pixel_diff > pixel_threshold) or (embed_diff > embed_threshold)
        if is_boundary:
            scenes.append({"scene_id":scene_idx,"start":round(float(scene_start),2),
                           "end":round(float(t),2),"duration":round(float(t)-float(scene_start),2),
                           "pixel_diff":round(pixel_diff,4),"semantic_diff":round(embed_diff,4)})
            boundaries.append({"t":round(float(t),2),"pixel_diff":round(pixel_diff,4),
                                "semantic_diff":round(embed_diff,4),
                                "type":"hard_cut" if pixel_diff>0.5 else "soft_transition"})
            scene_start=float(t); scene_idx+=1
        prev_arr=arr; prev_gray=gray
    if frames_ts:
        last_t=frames_ts[-1][0]
        if float(last_t)>float(scene_start):
            scenes.append({"scene_id":scene_idx,"start":round(float(scene_start),2),
                           "end":round(float(last_t),2),"duration":round(float(last_t)-float(scene_start),2)})
    return {"scenes":scenes,"boundaries":boundaries,"n_scenes":len(scenes),
            "n_boundaries":len(boundaries),
            "avg_scene_duration":round(float(np.mean([s["duration"] for s in scenes])) if scenes else 0.0,2)}

# ═════════════════════════════════════════════════════════
#  MODULE 4: EMOTION DETECTION  (FIXED — auto-detect arch)
# ═════════════════════════════════════════════════════════

EMOTION_LABELS = ["neutral","happy","sad","angry","surprised","fearful","disgusted","contempt"]

# ── Architecture A: RGB 3-channel 64×64 (default / fallback) ──────
class EmotionClassifierRGB(nn.Module):
    """
    3-channel RGB input, 64×64.
    Used when the .pth first conv weight has shape [32, 3, 3, 3].
    """
    def __init__(self, n_classes=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, n_classes),
        )
    def forward(self, x): return self.net(x)

# ── Architecture B: Grayscale 1-channel 48×48 (standard FER2013) ──
class EmotionClassifierGray(nn.Module):
    """
    1-channel grayscale input, 48×48.
    Used when the .pth first conv weight has shape [32, 1, 3, 3].
    This is the most common architecture for models trained on FER2013 / AffectNet.
    """
    def __init__(self, n_classes=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, n_classes),
        )
    def forward(self, x): return self.net(x)

# ── Architecture C: 7-class variant (FER2013 standard labels) ────
class EmotionClassifierRGB7(nn.Module):
    """
    3-channel RGB, 7 classes (no 'contempt').
    Used when .pth last linear weight has shape [7, 128].
    """
    def __init__(self, n_classes=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, n_classes),
        )
    def forward(self, x): return self.net(x)

class EmotionClassifierGray7(nn.Module):
    """
    1-channel grayscale, 7 classes.
    Used when .pth first conv is [32,1,3,3] and last linear is [7,128].
    """
    def __init__(self, n_classes=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, n_classes),
        )
    def forward(self, x): return self.net(x)

# Standard 7-class FER labels (no contempt)
EMOTION_LABELS_7 = ["angry","disgust","fear","happy","sad","surprise","neutral"]

def _load_resnet_emotion(state_dict, n_classes, device):
    from torchvision import models as tvm
    # Detect resnet18 vs resnet34 by presence of layer1.2 (3rd block = resnet34)
    is_34 = any("layer1.2" in k for k in state_dict.keys())
    model = tvm.resnet34(weights=None) if is_34 else tvm.resnet18(weights=None)
    arch  = "resnet34" if is_34 else "resnet18"
    model.fc = nn.Linear(512, n_classes)
    model.load_state_dict(state_dict, strict=True)
    print(f"[Emotion] ✓ {arch} detected and loaded")
    return model.to(device).eval()

def _detect_emotion_arch(state_dict: dict):
    keys = list(state_dict.keys())
    if not keys:
        return None, 8, 3, 64

    first_conv_key = next((k for k in keys if "weight" in k and len(state_dict[k].shape) == 4), None)
    linear_keys    = [k for k in keys if "weight" in k and len(state_dict[k].shape) == 2]
    last_linear_key = linear_keys[-1] if linear_keys else None

    in_channels = int(state_dict[first_conv_key].shape[1]) if first_conv_key else 3
    n_classes   = int(state_dict[last_linear_key].shape[0]) if last_linear_key else 8
    kernel_size = int(state_dict[first_conv_key].shape[2]) if first_conv_key else 3

    if kernel_size == 7:
        return "resnet18", n_classes, in_channels, 224
    elif in_channels == 1 and n_classes == 8:
        return EmotionClassifierGray,  8, 1, 48
    elif in_channels == 1 and n_classes == 7:
        return EmotionClassifierGray7, 7, 1, 48
    elif in_channels == 3 and n_classes == 7:
        return EmotionClassifierRGB7,  7, 3, 64
    else:
        return EmotionClassifierRGB,   8, 3, 64
# ── Global emotion state ───────────────────────────────────
_emotion_clf      = None    # loaded model
_emotion_in_ch    = 3       # detected: 1 = grayscale, 3 = RGB
_emotion_img_size = 64      # detected: 48 or 64
_emotion_n_cls    = 8       # detected: 7 or 8
_active_labels    = EMOTION_LABELS  # set at load time

def _make_emotion_transform(in_channels: int, img_size: int) -> transforms.Compose:
    """Build the correct preprocessing transform for the detected architecture."""
    if in_channels == 1:
        # Grayscale model: convert to L, then stack to 1-channel tensor
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),                          # → [1, H, W] in [0,1]
            transforms.Normalize([0.5], [0.5]),             # → [-1, 1]
        ])
    else:
        # RGB model
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),                          # → [3, H, W]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

_emotion_tf = _make_emotion_transform(3, 64)   # default; overwritten after load

def _get_emotion_clf():
    """
    Load EmotionClassifier with auto-detected architecture.

    Priority of checkpoint paths (first found wins):
      1. {WORK}/models/emotion_model.pth          ← your ByteStorm path
      2. models/emotion_model.pth                 ← relative (CWD)
      3. {WORK}/videomind/emotion_clf.pth         ← legacy name

    On first call the state_dict is inspected to determine:
      - input channels  (1 = grayscale / 3 = RGB)
      - n_classes       (7 or 8)
      - image size      (48 or 64)
    The correct architecture class is then instantiated and weights loaded.
    If no checkpoint is found, a random-weight RGB-8 model is used (inference will
    be random but the pipeline won't crash).
    """
    global _emotion_clf, _emotion_tf, _emotion_in_ch, _emotion_img_size
    global _emotion_n_cls, _active_labels

    if _emotion_clf is not None:
        return _emotion_clf

    ckpt_paths = [
        Path(f"{WORK}/models/emotion_model.pth"),
        Path("models/emotion_model.pth"),
        Path(f"{WORK}/videomind/emotion_clf.pth"),
    ]

    state_dict = None
    found_ckpt = None
    for ckpt in ckpt_paths:
        if ckpt.exists():
            try:
                state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)
                found_ckpt = ckpt
                print(f"[Emotion] ✓ Found checkpoint: {ckpt}")
                break
            except Exception as e:
                print(f"[Emotion] ⚠ Could not load {ckpt}: {e}")

    if state_dict is None:
        print("[Emotion] ⚠ No checkpoint found — using random-weight RGB-64 model.")
        print("[Emotion]   Place emotion_model.pth in:  {WORK}/models/  or  models/")
        clf = EmotionClassifierRGB(n_classes=8).to(GPU_EMOTION).eval()
        _emotion_clf      = clf
        _emotion_in_ch    = 3
        _emotion_img_size = 64
        _emotion_n_cls    = 8
        _active_labels    = EMOTION_LABELS
        _emotion_tf       = _make_emotion_transform(3, 64)
        return _emotion_clf

    # ── Auto-detect architecture ──────────────────────────
    ModelCls, n_cls, in_ch, img_sz = _detect_emotion_arch(state_dict)
    print(f"[Emotion] Detected arch → in_channels={in_ch}, n_classes={n_cls}, img_size={img_sz}")

    if ModelCls == "resnet18":
        clf = _load_resnet_emotion(state_dict, n_cls, GPU_EMOTION)
        _emotion_clf      = clf
        _emotion_in_ch    = in_ch
        _emotion_img_size = 224
        _emotion_n_cls    = n_cls
        _active_labels    = EMOTION_LABELS_7 if n_cls == 7 else EMOTION_LABELS
        _emotion_tf       = _make_emotion_transform(in_ch, 224)
        print(f"[Emotion] ✓ ResNet18 loaded. Labels: {_active_labels}")
        return _emotion_clf

    clf = ModelCls(n_classes=n_cls).to(GPU_EMOTION)

    try:
        clf.load_state_dict(state_dict, strict=True)
        print(f"[Emotion] ✓ Weights loaded (strict) from {found_ckpt}")
    except RuntimeError as e:
        print(f"[Emotion] ⚠ Strict load failed ({e})")
        try:
            # Try non-strict — handles minor key mismatches
            missing, unexpected = clf.load_state_dict(state_dict, strict=False)
            print(f"[Emotion] ✓ Weights loaded (non-strict). "
                  f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            if missing:
                print(f"[Emotion]   Missing keys  : {missing[:5]}")
            if unexpected:
                print(f"[Emotion]   Unexpected keys: {unexpected[:5]}")
        except Exception as e2:
            print(f"[Emotion] ✗ Non-strict load also failed: {e2}")
            print("[Emotion]   Falling back to random-weight model.")

    clf.eval()

    # ── Update module-level state ─────────────────────────
    _emotion_clf      = clf
    _emotion_in_ch    = in_ch
    _emotion_img_size = img_sz
    _emotion_n_cls    = n_cls
    _active_labels    = EMOTION_LABELS_7 if n_cls == 7 else EMOTION_LABELS
    _emotion_tf       = _make_emotion_transform(in_ch, img_sz)

    print(f"[Emotion] Active labels: {_active_labels}")
    return _emotion_clf

# ── Warm up at import time ────────────────────────────────
try:
    _get_emotion_clf()
except Exception as e:
    print(f"[Emotion] Init error: {e}")

_emotion_history: List[str] = []

def detect_emotion_frame(img: Image.Image, t: float) -> dict:
    faces = extract_faces(img)
    if not faces:
        # Try center crop as fallback before giving up
        w, h = img.size
        pad = min(w, h) // 4
        center_crop = img.crop((pad, pad, w-pad, h-pad))
        faces = [center_crop]

    clf = _get_emotion_clf()
    frame_emotions = []

    for face in faces[:2]:
        try:
            # _emotion_tf is set correctly (grayscale or RGB) at load time
            ft = _emotion_tf(face).unsqueeze(0).to(GPU_EMOTION)
            with torch.no_grad():
                logits = clf(ft)
            probs   = F.softmax(logits, dim=-1)[0].cpu().numpy()
            top_idx = int(np.argmax(probs))
            all_sc  = {_active_labels[i]: round(float(probs[i]), 4)
                       for i in range(len(_active_labels))}
            frame_emotions.append((_active_labels[top_idx], float(probs[top_idx]), all_sc))
            del ft
        except Exception as ex:
            print(f"[Emotion] Frame inference error: {ex}")
            continue

    if not frame_emotions:
        return {"t": round(t, 2), "emotion": "unknown", "confidence": 0.0,
                "face_count": len(faces), "all_scores": {}}

    primary_emotion, primary_conf, all_scores = frame_emotions[0]

    # Temporal smoothing over last 3 frames
    _emotion_history.append(primary_emotion)
    if len(_emotion_history) > 3:
        _emotion_history.pop(0)
    smoothed = Counter(_emotion_history).most_common(1)[0][0]

    return {"t": round(t, 2), "emotion": smoothed, "raw_emotion": primary_emotion,
            "confidence": round(primary_conf, 4), "face_count": len(faces),
            "all_scores": all_scores}

def analyze_emotion_timeline(frames_ts, keyframe_interval=5):
    timeline = []
    emotion_dist = {e: 0 for e in _active_labels}
    emotion_dist["no_face"] = 0
    _emotion_history.clear()

    for idx, (t, img) in enumerate(frames_ts):
        if idx % keyframe_interval != 0:
            continue
        result = detect_emotion_frame(img, t)
        timeline.append(result)
        em = result.get("emotion", "no_face")
        if em in emotion_dist:
            emotion_dist[em] += 1
        else:
            emotion_dist["no_face"] += 1

    torch.cuda.empty_cache()
    total    = max(sum(emotion_dist.values()), 1)
    dominant = max(emotion_dist, key=emotion_dist.get)
    return {
        "timeline":            timeline,
        "dominant_emotion":    dominant,
        "emotion_distribution": {k: round(v / total, 4) for k, v in emotion_dist.items() if v > 0},
        "n_frames_analyzed":   len(timeline),
        "model_info": {
            "in_channels":  _emotion_in_ch,
            "img_size":     _emotion_img_size,
            "n_classes":    _emotion_n_cls,
            "labels":       _active_labels,
        },
    }

# ═════════════════════════════════════════════════════════
#  MODULE 5: rPPG PHYSIOLOGICAL
# ═════════════════════════════════════════════════════════

def estimate_rppg_pulse(frames_ts, fps=25.0):
    face_signals = []
    for t, img in frames_ts:
        faces = extract_faces(img)
        if not faces: face_signals.append(None); continue
        face = faces[0]; arr = np.array(face.resize((64,64))).astype(np.float32)
        forehead = arr[:int(0.30*arr.shape[0]),:,:]
        face_signals.append(float(np.mean(forehead[:,:,1])))
    valid_signals = [s for s in face_signals if s is not None]
    if len(valid_signals) < 16:
        return {"pulse_bpm":0.0,"pulse_confidence":0.0,"liveness_score":0.5,
                "method":"insufficient_data","n_face_frames":len(valid_signals)}
    sig = np.array(valid_signals, dtype=np.float64) - np.mean(np.array(valid_signals))
    n   = len(sig); freqs = np.fft.rfftfreq(n, d=1.0/fps); fft_mag = np.abs(np.fft.rfft(sig))
    band_mask = (freqs >= 0.7) & (freqs <= 4.0)
    if not band_mask.any():
        return {"pulse_bpm":0.0,"pulse_confidence":0.0,"liveness_score":0.5,"method":"no_band_signal"}
    band_mag  = fft_mag * band_mask; peak_idx = int(np.argmax(band_mag))
    peak_freq = float(freqs[peak_idx]); peak_power = float(band_mag[peak_idx])
    total_power = float(np.sum(fft_mag)+1e-9)
    pulse_bpm = round(peak_freq*60.0,1)
    pulse_confidence = round(float(np.clip(peak_power/total_power*10,0,1)),4)
    is_plausible = 42 <= pulse_bpm <= 150
    liveness = round(float(np.clip(pulse_confidence*(1.0 if is_plausible else 0.2),0,1)),4)
    return {"pulse_bpm":pulse_bpm,"pulse_confidence":pulse_confidence,"liveness_score":liveness,
            "plausible_hr":is_plausible,"n_face_frames":len(valid_signals),"method":"rppg_fft"}

# ═════════════════════════════════════════════════════════
#  MODULE 6: AUDIO ACTIVITY
# ═════════════════════════════════════════════════════════

def analyze_audio_activity(audio_path, segments):
    if HAS_LIBROSA:
        try:
            y, sr    = librosa.load(audio_path, sr=None, mono=True)
            duration = float(len(y))/sr
            frame_len= int(sr); hop_len=int(sr//2); rms_frames=[]
            for start in range(0, len(y)-frame_len, hop_len):
                chunk=y[start:start+frame_len]; rms=float(np.sqrt(np.mean(chunk**2)))
                rms_frames.append({"t":round(float(start)/sr,2),"rms":round(rms,6)})
            rms_vals  = np.array([f["rms"] for f in rms_frames])
            active_pct= float(np.mean(rms_vals > float(np.mean(rms_vals)*0.5)))
            centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)
            zcr       = librosa.feature.zero_crossing_rate(y)
            mfccs     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfcc_var  = float(np.mean(np.var(mfccs, axis=1)))
            try:
                f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=600, sr=sr, frame_length=2048)
                vf0 = f0[voiced] if voiced is not None else np.array([])
                pitch_reg = float(1.0-np.clip(np.std(vf0)/50.0,0,1)) if len(vf0)>10 else 0.5
            except Exception: pitch_reg = 0.5
            return {"energy_timeline":rms_frames[:200],"speech_activity":round(active_pct,4),
                    "mean_rms":round(float(np.mean(rms_vals)),6),"peak_rms":round(float(np.max(rms_vals)),6),
                    "spectral_centroid":round(float(np.mean(centroid)),2),
                    "zero_crossing_rate":round(float(np.mean(zcr)),6),
                    "mfcc_variance":round(mfcc_var,4),"pitch_regularity":round(pitch_reg,4),
                    "clone_score":round(float(np.clip(pitch_reg*0.8,0,1)),4),
                    "duration_analyzed":round(duration,2),"method":"librosa_full"}
        except Exception as e:
            print(f"[AudioActivity] {e}")
    if not segments:
        return {"speech_activity":0.0,"method":"no_data"}
    total_speech = sum(max(0,s["end"]-s["start"]) for s in segments)
    all_text     = " ".join(s.get("text","") for s in segments)
    word_rate    = len(all_text.split())/max(total_speech,1.0)
    return {"speech_activity":round(min(total_speech/max(segments[-1]["end"] if segments else 1.0,1.0),1.0),4),
            "word_rate_per_sec":round(word_rate,2),"total_speech_sec":round(total_speech,2),
            "clone_score":0.3,"method":"segment_fallback"}

# ═════════════════════════════════════════════════════════
#  MODULE 7: MULTIMODAL ALIGNMENT
# ═════════════════════════════════════════════════════════

def compute_multimodal_alignment_clip(frames_ts, segments):
    proc, model = get_clip()
    aligned_scores=[]; misaligned=[]; misalign_windows=[]
    for seg in segments[:30]:
        text = seg.get("text","").strip()
        if not text: continue
        seg_t = float(seg.get("start",0))
        best_img=None; best_dt=float("inf")
        for t, img in frames_ts:
            dt = abs(float(t)-seg_t)
            if dt < best_dt and dt <= 3.0: best_dt=dt; best_img=img
        if best_img is None: continue
        try:
            inputs = proc(text=[text], images=best_img, return_tensors="pt", padding=True).to(GPU_CLIP)
            with torch.no_grad():
                out = model(**inputs)
            sim = float(F.cosine_similarity(out.text_embeds, out.image_embeds).item())
            aligned_scores.append({"t":round(seg_t,2),"text":text[:80],"clip_sim":round(sim,4),
                                   "aligned":sim>0.20,"dt_sec":round(best_dt,2)})
            if sim < 0.15:
                misaligned.append({"t":round(seg_t,2),"text":text[:60],"clip_sim":round(sim,4)})
        except Exception: continue
    if not aligned_scores:
        return {"mean_clip_sim":0.5,"method":"insufficient_data","aligned_segments":[],
                "misaligned_segments":[],"misalignment_windows":[]}
    sims    = [s["clip_sim"] for s in aligned_scores]
    sim_arr = np.array(sims); variance = float(np.var(sim_arr))
    in_win=False; win_start=None
    for s in aligned_scores:
        if not s["aligned"]:
            if not in_win: in_win=True; win_start=s["t"]
        else:
            if in_win:
                misalign_windows.append({"start":win_start,"end":s["t"],
                                         "reason":"Consecutive audio-visual misalignment window"})
                in_win=False
    if in_win and win_start is not None:
        misalign_windows.append({"start":win_start,"end":aligned_scores[-1]["t"],
                                  "reason":"Consecutive audio-visual misalignment window (open)"})
    torch.cuda.empty_cache()
    return {"mean_clip_sim":round(float(np.mean(sims)),4),"min_clip_sim":round(float(np.min(sims)),4),
            "max_clip_sim":round(float(np.max(sims)),4),"alignment_variance":round(variance,6),
            "alignment_rate":round(float(np.mean([s["aligned"] for s in aligned_scores])),4),
            "aligned_segments":aligned_scores,"misaligned_segments":misaligned,
            "misalignment_windows":misalign_windows,"n_analyzed":len(aligned_scores),
            "method":"clip_fine_grained"}

# ═════════════════════════════════════════════════════════
#  STAGE 1 — CLIP + VLM  (used in all modes)
# ═════════════════════════════════════════════════════════

CLIP_LABELS = [
    "a real human face","natural outdoor scene",
    "computer-generated graphics","animated content","news broadcast",
    "interview or talking head","action scene","text on screen",
]
SCENE_CAPTION_PROMPT = (
    "Describe this video frame briefly. What is happening? "
    "Who or what is visible? One or two sentences only."
)
FORENSIC_PROMPT = (
    "Examine this frame for visual artifacts, unnatural textures, "
    "lighting inconsistencies, or signs of AI generation. "
    "End with: VERDICT: REAL or VERDICT: AI_GENERATED"
)

def clip_embed_frame(img, cache_key=None):
    if cache_key:
        cached = feature_store.get(cache_key)
        if cached is not None: return cached
    proc, model = get_clip()
    with torch.no_grad():
        feat = model.get_image_features(**proc(images=img, return_tensors="pt").to(GPU_CLIP))
    if not isinstance(feat, torch.Tensor):
        feat = feat.pooler_output if hasattr(feat, 'pooler_output') else feat.last_hidden_state[:,0,:]
    if not isinstance(feat, __import__("torch").Tensor):
        feat = feat.pooler_output if hasattr(feat, "pooler_output") else feat.last_hidden_state[:,0,:]
    feat = feat / feat.norm(dim=-1, keepdim=True)
    arr  = feat.cpu().numpy()[0]
    if cache_key: feature_store.put(cache_key, arr)
    return arr

def clip_scene_classify(img):
    proc, model = get_clip()
    with torch.no_grad():
        logits = model(**proc(text=CLIP_LABELS, images=img,
                              return_tensors="pt", padding=True).to(GPU_CLIP)).logits_per_image[0]
    probs = logits.softmax(0).cpu().numpy()
    i = int(probs.argmax())
    return {"label":CLIP_LABELS[i],"confidence":round(float(probs[i]),4)}

def qwen_caption(img, forensic=False):
    prompt = FORENSIC_PROMPT if forensic else SCENE_CAPTION_PROMPT
    result = get_qwen()
    try:
        proc, model = result
        if hasattr(proc,"apply_chat_template"):
            msgs   = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":prompt}]}]
            text   = proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
            inputs = proc(text=text,images=[img],return_tensors="pt").to(GPU_QWEN)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=VLM_MAX_TOKENS)
            raw = proc.decode(out[0],skip_special_tokens=True).strip()
            # Strip prompt leakthrough
            if "assistant" in raw.lower():
                raw = raw.split("assistant")[-1].strip().lstrip(":").strip()
            return raw
        else:
            inputs = proc(images=img,return_tensors="pt").to(GPU_QWEN)
            with torch.no_grad():
                tokens = model.generate(**inputs, max_new_tokens=60)
            return proc.tokenizer.decode(tokens[0],skip_special_tokens=True)
    except Exception as e:
        return f"Caption unavailable ({e})"

def run_stage1_video_to_text(frames_ts, video_hash, progress_cb, forensic_mode=False):
    progress_cb(2, "Stage 1 — CLIP embeddings + captions...")
    clip_embs=[]; scene_labels=[]; captions=[]; keyframe_forensics=[]
    for idx,(t,img) in enumerate(frames_ts):
        try:
            ck  = f"{video_hash}_clip_{idx}"
            emb = clip_embed_frame(img, cache_key=ck)
            clip_embs.append(emb)
            scene = clip_scene_classify(img)
            scene_labels.append({"t":float(t),**scene})
            is_kf = (idx % KEYFRAME_INTERVAL == 0)
            if is_kf:
                caption  = qwen_caption(img, forensic=False)
                captions.append({"t":float(t),"caption":caption,"keyframe":True})
                if forensic_mode:
                    forensic_cap = qwen_caption(img, forensic=True)
                    keyframe_forensics.append({"t":float(t),"narrative":forensic_cap})
            else:
                captions.append({"t":float(t),"caption":scene["label"],"keyframe":False})
        except Exception as _frame_err:
            print(f"[Stage1] Frame {idx} @ t={t:.1f}s error: {_frame_err}")
            captions.append({"t":float(t),"caption":"unavailable","keyframe":False})
            continue
    forensic_flags=[]; ai_frame_ratio=0.0
    if forensic_mode:
        for kf in keyframe_forensics:
            text  = kf["narrative"].upper()
            label = "AI_GENERATED" if "AI_GENERATED" in text else "REAL"
            forensic_flags.append({"t":kf["t"],"vlm_verdict":label})
        ai_frame_ratio = (sum(1 for f in forensic_flags if f["vlm_verdict"]=="AI_GENERATED")
                          / max(len(forensic_flags),1))
    torch.cuda.empty_cache()
    progress_cb(2,"Stage 1 ✓")
    return {"clip_embs":clip_embs,"scene_labels":scene_labels,"captions":captions,
            "keyframe_forensics":keyframe_forensics,"forensic_flags":forensic_flags,
            "ai_frame_ratio":round(ai_frame_ratio,4),"n_keyframes":len(keyframe_forensics)}

# ═════════════════════════════════════════════════════════
#  STAGE 2 — WHISPER
# ═════════════════════════════════════════════════════════

def run_stage2_audio_to_text(audio_path, progress_cb):
    progress_cb(3,"Stage 2 — Whisper transcription...")
    try:
        wm  = get_whisper()
        raw = wm.transcribe(audio_path, verbose=False)
        transcript = raw.get("text","").strip()
        segments   = [{"start":float(s.get("start",0.0)),"end":float(s.get("end",0.0)),
                       "text":(s.get("text") or "").strip()}
                      for s in raw.get("segments",[])]
        torch.cuda.empty_cache()
        progress_cb(3,"Stage 2 ✓")
        return {"transcript":transcript,"segments":segments,
                "language":raw.get("language","unknown"),"word_count":len(transcript.split())}
    except Exception as e:
        progress_cb(3,"Stage 2 — Completing audio analysis...")
        return {"transcript":"","segments":[],"language":"unknown","word_count":0}

# ═════════════════════════════════════════════════════════
#  UNIFIED REPRESENTATION
# ═════════════════════════════════════════════════════════

def build_unified_representation(stage1, stage2, duration):
    segments = sorted(stage2["segments"], key=lambda x: x["start"])
    captions = sorted(stage1.get("captions", []), key=lambda x: x["t"])
    n_caps   = len(captions); unified=[]
    for seg in segments:
        s,e,text = seg["start"],seg["end"],seg["text"]
        nearby_caps=[]
        for ci in range(n_caps):
            ct=captions[ci]["t"]
            cap = captions[ci]["caption"]
            if s-2.0<=ct<=e+2.0 and cap and "apply_chat_template" not in cap and "unavailable" not in cap.lower():
                nearby_caps.append(cap)
        seen,unique_caps=set(),[]
        for c in nearby_caps:
            k=c.strip().lower()
            if k and k not in seen: seen.add(k); unique_caps.append(c)
        combined = (text+" [Visual: "+" | ".join(unique_caps[:2])+"]" if unique_caps else text)
        unified.append({"start":s,"end":e,"text":text,"visual_caps":unique_caps[:2],"combined_text":combined})
    return {"unified_segments":unified,"full_text":" ".join(u["combined_text"] for u in unified),"duration":duration}

# ═════════════════════════════════════════════════════════
#  STAGE 3 — DRE + AV SYNC
# ═════════════════════════════════════════════════════════

class LightUNet(nn.Module):
    def __init__(self, ch=3, base=32):
        super().__init__()
        def blk(i,o): return nn.Sequential(
            nn.Conv2d(i,o,3,padding=1),nn.GroupNorm(8,o),nn.SiLU(),
            nn.Conv2d(o,o,3,padding=1),nn.GroupNorm(8,o),nn.SiLU())
        self.e1=blk(ch,base);self.e2=blk(base,base*2);self.e3=blk(base*2,base*4)
        self.mid=blk(base*4,base*4);self.d3=blk(base*8,base*2);self.d2=blk(base*4,base)
        self.d1=blk(base*2,base);self.out=nn.Conv2d(base,ch,1)
        self.pool=nn.MaxPool2d(2);self.up=nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False)
    def forward(self,x):
        e1=self.e1(x);e2=self.e2(self.pool(e1));e3=self.e3(self.pool(e2))
        m=self.mid(self.pool(e3));d3=self.d3(torch.cat([self.up(m),e3],1))
        d2=self.d2(torch.cat([self.up(d3),e2],1));d1=self.d1(torch.cat([self.up(d2),e1],1))
        return self.out(d1)

_unet = LightUNet().to(GPU_FUSION).eval()
_unet_ckpt = Path(f"{WORK}/videomind/unet_denoiser.pth")
_unet_trained = _unet_ckpt.exists()
if _unet_trained:
    _unet.load_state_dict(torch.load(_unet_ckpt, map_location=GPU_FUSION))
    print("[DRE] ✓ UNet checkpoint loaded")
else:
    print("[DRE] ⚠ No UNet checkpoint — DRE scores will be neutral (0.5).")

_dre_tf = transforms.Compose([
    transforms.Resize((128,128)),transforms.ToTensor(),transforms.Normalize([0.5]*3,[0.5]*3)])

def compute_dre(img):
    if not _unet_trained:
        return 0.5
    x0=_dre_tf(img).unsqueeze(0).to(GPU_FUSION); noise=torch.randn_like(x0)*0.5
    with torch.no_grad(): x_hat=_unet(x0+noise)
    score=round(float(np.clip(F.mse_loss(x_hat,x0).item()/0.25,0.0,1.0)),4)
    del x0,noise,x_hat; return score

MOUTH_ROI=(0.30,0.70,0.55,0.95)

def compute_av_sync(frames, segments, duration):
    if len(frames)<4: return {"av_offset_ms":0.0,"sync_score":0.5}
    signal=[]; prev=None
    for img in frames:
        arr=np.array(img.resize((128,128))); H,W=arr.shape[:2]
        x1,x2,y1,y2=MOUTH_ROI
        roi=arr[int(y1*H):int(y2*H),int(x1*W):int(x2*W)]
        gray=np.mean(roi,axis=2).astype(np.float32)
        if prev is not None:
            diff=float(np.abs(gray-prev).mean())
            if HAS_CV2:
                try:
                    flow=cv2.calcOpticalFlowFarneback(prev,gray,None,0.5,3,15,3,5,1.2,0)
                    diff=float(np.sqrt((flow**2).sum(-1)).mean()); del flow
                except Exception: pass
            signal.append(diff)
        prev=gray
    if not signal: return {"av_offset_ms":0.0,"sync_score":0.5}
    n=len(signal)
    # time_per_sample = real seconds between each signal sample
    time_per_sample=duration/max(len(frames),1)
    env=np.zeros(n,dtype=np.float32)
    for seg in segments:
        # convert segment times to signal indices using time_per_sample
        sf=int(seg["start"]/time_per_sample)
        ef=int(seg["end"]/time_per_sample)
        e=min(len(seg["text"].strip())/50.0,1.0)
        for i in range(max(0,sf),min(n,ef+1)): env[i]=e
    def norm(s): s=s-s.mean(); return s/(s.std()+1e-9)
    # Only run xcorr if both signals have variance
    if np.std(signal)<1e-6 or np.std(env)<1e-6:
        return {"av_offset_ms":0.0,"sync_score":1.0}
    xcorr=np.correlate(norm(np.array(signal)),norm(env[:n]),mode="full")
    lag=int(xcorr.argmax())-(n-1)
    off=round(lag*time_per_sample*1000,2)
    off=float(np.clip(off,-2000,2000))
    return {"av_offset_ms":off,"sync_score":round(float(np.clip(1.0-abs(off)/2000.0,0,1)),4)}

def check_text_video_alignment(clip_embs, unified, embed_model):
    if not clip_embs or not unified["full_text"].strip():
        return {"alignment_score":0.5,"visual_coherence":0.5,"text_coherence":0.5,"method":"insufficient_data"}
    if len(clip_embs)>=2:
        sims=[float(cosine_similarity(clip_embs[i][None],clip_embs[i+1][None])[0,0])
              for i in range(len(clip_embs)-1)]
        visual_coherence=float(np.mean(sims))
    else: visual_coherence=1.0
    texts=[s["text"] for s in unified["unified_segments"] if s["text"].strip()][:20]
    if len(texts)>=2:
        te=embed_model.encode(texts)
        text_coherence=float(np.mean([float(cosine_similarity(te[i:i+1],te[i+1:i+2])[0,0])
                                       for i in range(len(te)-1)]))
    else: text_coherence=0.5
    return {"alignment_score":round((visual_coherence*0.5+text_coherence*0.5),4),
            "visual_coherence":round(visual_coherence,4),"text_coherence":round(text_coherence,4),
            "visual_coherence":round(visual_coherence,4),"text_coherence":round(text_coherence,4),"method":"cosine_frame_text_coherence"}

def run_stage3_text_to_video(frames_ts, stage1, stage2, unified, progress_cb):
    progress_cb(4,"Stage 3 — DRE + AV-sync...")
    frames=[img for _,img in frames_ts]
    dre_scores=[{"t":float(t),"ood_score":compute_dre(img)} for t,img in frames_ts[::4]]
    mean_dre=round(float(np.mean([d["ood_score"] for d in dre_scores])),4) if dre_scores else 0.0
    max_dre =round(float(np.max ([d["ood_score"] for d in dre_scores])),4) if dre_scores else 0.0
    av_sync    = compute_av_sync(frames, stage2["segments"], unified["duration"])
    # Clamp AV offset — cross-correlation can produce garbage on short videos
    if abs(av_sync.get("av_offset_ms", 0)) > 500:
        av_sync["av_offset_ms"] = 0.0
        av_sync["sync_score"] = 0.85
    embed_model= get_embed()
    alignment  = check_text_video_alignment(stage1.get("clip_embs", []), unified, embed_model)
    embs=stage1.get("clip_embs", []); tc=1.0
    if len(embs)>=2:
        tc=round(float(np.mean([float(cosine_similarity(embs[i][None],embs[i+1][None])[0,0])
                                  for i in range(len(embs)-1)])),4)
    torch.cuda.empty_cache(); progress_cb(4,"Stage 3 ✓")
    return {"dre_scores":dre_scores,"dre_mean":mean_dre,"dre_max":max_dre,
            "av_sync":av_sync,"alignment":alignment,"temporal_consistency":tc}

# ═════════════════════════════════════════════════════════
#  STAGE 4 — SparseMoE FUSION
# ═════════════════════════════════════════════════════════

class ContextAwareTemporalMoE(nn.Module):
    TOP_K=2; N_EXPERTS=4; IN_DIM=16; CTX_DIM=12
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(self.IN_DIM),nn.Linear(self.IN_DIM,32),nn.GELU(),
                          nn.Dropout(0.1),nn.Linear(32,3))
            for _ in range(self.N_EXPERTS)])
        self.temporal_attn = nn.MultiheadAttention(embed_dim=self.IN_DIM,num_heads=2,
                                                    batch_first=True,dropout=0.05)
        self.gate = nn.Sequential(nn.Linear(self.CTX_DIM,24),nn.GELU(),nn.Linear(24,self.N_EXPERTS))
        self.calib = nn.Sequential(nn.Linear(3,3),nn.Softmax(dim=-1))
    def forward(self, features, context):
        attn_out,_=self.temporal_attn(features,features,features); features=features+attn_out
        gate_logits=self.gate(context); topk_v,topk_i=torch.topk(gate_logits,self.TOP_K,dim=-1)
        sparse_w=torch.zeros_like(gate_logits); sparse_w.scatter_(1,topk_i,F.softmax(topk_v,-1))
        e_outs=torch.stack([self.experts[i](features[:,i,:]) for i in range(self.N_EXPERTS)],dim=1)
        combined=(e_outs*sparse_w.unsqueeze(-1)).sum(1); probs=self.calib(combined)
        load=sparse_w.mean(0); cv2_val=load.var()/(load.mean()**2+1e-9)
        return probs,topk_i,sparse_w,cv2_val

_moe = ContextAwareTemporalMoE().to(GPU_FUSION).eval()
_moe_ckpt = Path(f"{WORK}/videomind/sparse_moe_vm.pth")
if _moe_ckpt.exists():
    try:
        ckpt = torch.load(_moe_ckpt, map_location=GPU_FUSION, weights_only=False)
        # Unwrap if saved as a training checkpoint dict
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            ckpt = ckpt["model_state"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        _moe_raw = ckpt; ckpt = _moe_raw.get('model_state', _moe_raw) if isinstance(_moe_raw, dict) else _moe_raw
        missing, unexpected = _moe.load_state_dict(ckpt, strict=False)
        print(f'[MoE] Loaded missing={len(missing)} unexpected={len(unexpected)}')
        print("[VideoMind] ✓ MoE loaded")
    except Exception as e:
        print(f"[VideoMind] ⚠ MoE checkpoint incompatible: {e}")

def run_stage4_fusion(stage1, stage2, stage3, reference_text, temporal_results, physiological, audio_activity, progress_cb, deepfake_results=None):
    progress_cb(5,"Stage 4 — MoE fusion...")
    tc=stage3["temporal_consistency"]; align=stage3["alignment"]
    av_sync=stage3["av_sync"]; dre_mean=stage3["dre_mean"]; ai_ratio=stage1.get("ai_frame_ratio", 0)
    ai_label_count=sum(1 for s in stage1.get("scene_labels", []) if "AI" in s["label"].upper())/max(len(stage1.get("scene_labels", [])),1)
    av_fake=float(1.0-av_sync["sync_score"])
    clone_score=audio_activity.get("clone_score",0.3)
    ref_score=0.0
    if reference_text.strip() and stage2["transcript"].strip():
        em=get_embed()
        e1=em.encode([stage2["transcript"][:512]]); e2=em.encode([reference_text])
        ref_score=float(cosine_similarity(e1,e2)[0][0])
    e0=np.array([ai_ratio,ai_label_count,1.0-tc,align.get("visual_coherence", 0.5),0,0,0,0,0,0,0,0,0,0,0,0],dtype=np.float32)[:16]
    e1=np.array([dre_mean,stage3["dre_max"],1.0-tc,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype=np.float32)[:16]
    e2=np.array([av_fake,clone_score,abs(av_sync["av_offset_ms"])/2000.0,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype=np.float32)[:16]
    e3=np.array([1.0-align.get("alignment_score", 0.5),align.get("text_coherence", 0.5),ref_score,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype=np.float32)[:16]
    n_anomaly_segs=temporal_results.get("n_anomaly_segments",0)
    liveness=physiological.get("liveness_score",0.5)
    pitch_reg=audio_activity.get("pitch_regularity",0.5)
    ctx=np.array([dre_mean,ai_ratio,av_fake,1.0-align.get("alignment_score", 0.5),1.0-tc,ref_score,
                  float(stage2["word_count"])/500.0,align.get("visual_coherence", 0.5),
                  float(min(n_anomaly_segs,10))/10.0,liveness,align.get("alignment_score", 0.5),pitch_reg],dtype=np.float32)
    try:
        feats_t=torch.tensor(np.stack([e0,e1,e2,e3]),dtype=torch.float32).unsqueeze(0).to(GPU_FUSION)
        ctx_t=torch.tensor(ctx,dtype=torch.float32).unsqueeze(0).to(GPU_FUSION)
        with torch.no_grad():
            probs,active_idx,sparse_w,lb=_moe(feats_t,ctx_t)
        pl=probs[0].cpu().tolist()
        names=["visual_semantics","temporal_integrity","av_truth","cross_modal_alignment"]
        active=[names[i] for i in active_idx[0].tolist()]
        gate_w={n:round(float(sparse_w[0,i]),4) for i,n in enumerate(names)}
        labels=["human","mixed","ai_generated"]
        # CNN is the primary signal — MoE gets only 10% influence
        cnn_score = (deepfake_results or {}).get("mean_fake_score", 0.5)
        moe_score = pl[2] + 0.5*pl[1]   # MoE ai probability
        ai_score  = round(float(0.90*cnn_score + 0.10*moe_score), 4)
        if ai_score < 0.35:
            pred_idx = 0   # human
        elif ai_score > 0.60:
            pred_idx = 2   # ai_generated
        else:
            pred_idx = 1   # mixed
        print(f"[Verdict] cnn={cnn_score:.3f} moe={moe_score:.3f} final={ai_score:.3f} label={labels[pred_idx]}")
        fusion_result={"label":labels[pred_idx],"ai_score":round(float(ai_score),4),
                       "prob_human":round(pl[0],4),"prob_mixed":round(pl[1],4),"prob_ai":round(pl[2],4),
                       "active_experts":active,"gate_weights":gate_w,"method":"context_aware_temporal_moe"}
    except Exception as e:
        print(f"[Fusion] MoE error: {e}")
        ai_score=float(np.clip(0.3*dre_mean+0.3*ai_ratio+0.2*av_fake+0.2*(1-align.get("alignment_score", 0.5)),0,1))
        label="ai_generated" if ai_score>0.88 else ("human" if ai_score<0.55 else "mixed")
        fusion_result={"label":label,"ai_score":round(ai_score,4),"method":"heuristic"}
    torch.cuda.empty_cache(); progress_cb(5,"Stage 4 ✓")
    return {"fusion":fusion_result,"ref_score":round(ref_score,4),
            "expert_inputs":{"visual_semantics":e0[:4].tolist(),"temporal":e1[:3].tolist(),
                             "av_truth":e2[:3].tolist(),"cross_modal":e3[:3].tolist()}}

# ═════════════════════════════════════════════════════════
#  ANALYTICAL REPORT
# ═════════════════════════════════════════════════════════

def build_analytical_report(stage1,stage2,stage3,stage4,deepfake_results,temporal_results,
                             scene_results,emotion_results,audio_activity,mm_alignment,
                             physiological,unified,duration):
    fusion=stage4["fusion"]
    df_score=deepfake_results.get("mean_fake_score",0.0)
    temporal_risk=min(temporal_results.get("mean_anomaly",0.0)*2,1.0)
    audio_clone=audio_activity.get("clone_score",0.3)
    moe_ai_score=fusion.get("ai_score",0.0)
    clip_align_score=mm_alignment.get("mean_clip_sim",0.5)
    liveness_score=physiological.get("liveness_score",0.5)
    # OOD score from DRE UNet — lives in stage3
    dre_ood = float(stage3.get("dre_mean", 0.5)) if isinstance(stage3, dict) else 0.5
    composite_risk=round(float(np.clip(
        0.80*df_score+0.08*moe_ai_score+0.01*dre_ood+
        0.10*temporal_risk+
        0.10*(1.0-stage3["av_sync"]["sync_score"])+
        0.05*(1.0-liveness_score)+0.05*audio_clone,0.0,1.0)),4)
    def risk_level(s):
        if s<0.40: return "LOW"
        if s<0.65: return "MEDIUM"
        if s<0.85: return "HIGH"
        return "CRITICAL"
    flags=[]
    if df_score>0.55: flags.append({"type":"DEEPFAKE_FACE","severity":"HIGH","detail":f"CNN+freq fake score {df_score:.2f}"})
    if temporal_results.get("n_anomaly_segments",0)>1:
        segs=temporal_results.get("anomaly_segments",[])
        seg_str="; ".join(f"{s['start']}–{s['end']}s" for s in segs[:3])
        flags.append({"type":"TEMPORAL_SPLICE","severity":"MEDIUM","detail":f"Anomaly windows: {seg_str}"})
    if abs(stage3['av_sync']['av_offset_ms']) > 300:
        flags.append({"type":"AV_DESYNC","severity":"HIGH","detail":f"AV offset {stage3['av_sync']['av_offset_ms']:.0f}ms"})
    if stage1.get('ai_frame_ratio', 0) > 0.3:
        flags.append({"type":"AI_VISUAL_LABELS","severity":"MEDIUM","detail":f"{stage1['ai_frame_ratio']*100:.0f}% CLIP labels indicate AI"})
    if audio_clone>0.7:
        flags.append({"type":"VOICE_CLONE","severity":"HIGH","detail":f"Pitch regularity {audio_clone:.2f}"})
    if liveness_score<0.3 and physiological.get("method")=="rppg_fft":
        flags.append({"type":"NO_PULSE_DETECTED","severity":"HIGH","detail":f"rPPG liveness {liveness_score:.2f}"})
    if len(mm_alignment.get("misalignment_windows", [])) > 25:
        flags.append({"type":"MISALIGNMENT_WINDOWS","severity":"MEDIUM",
                      "detail":f"{len(mm_alignment['misalignment_windows'])} misalignment window(s)"})
    return {"composite_risk_score":composite_risk,"risk_level":risk_level(composite_risk),
            "overall_verdict":fusion["label"].upper(),"confidence":round(1.0-abs(0.5-composite_risk)*0.5,4),
            "evidence_flags":flags,
            "module_scores":{"deepfake_cnn_freq":round(df_score,4),"temporal_anomaly":round(temporal_risk,4),
                             "av_sync":round(stage3["av_sync"]["sync_score"],4),
                             "clip_alignment":round(clip_align_score,4),"moe_fusion":round(moe_ai_score,4),
                             "dre_reconstruction":round(stage3["dre_mean"],4),
                             "audio_clone":round(audio_clone,4),"rppg_liveness":round(liveness_score,4)},
            "scene_summary":{"n_scenes":scene_results.get("n_scenes",0),"n_cuts":scene_results.get("n_boundaries",0),
                             "avg_scene_duration":scene_results.get("avg_scene_duration",0.0)},
            "emotion_summary":{"dominant":emotion_results.get("dominant_emotion","unknown"),
                               "distribution":emotion_results.get("emotion_distribution",{})},
            "audio_summary":{"speech_activity":audio_activity.get("speech_activity",0.0),
                             "clone_score":audio_clone,"pitch_regularity":audio_activity.get("pitch_regularity",0.5)},
            "physiological_summary":physiological,"deepfake_summary":deepfake_results,
            "temporal_summary":{"max_anomaly":temporal_results.get("max_anomaly",0.0),
                                "n_suspicious":temporal_results.get("n_suspicious",0),
                                "anomaly_segments":temporal_results.get("anomaly_segments",[])},
            "alignment_summary":{"mean_clip_sim":mm_alignment.get("mean_clip_sim",0.5),
                                 "alignment_variance":mm_alignment.get("alignment_variance",0.0),
                                 "misalignment_windows":mm_alignment.get("misalignment_windows",[])}}

# ═════════════════════════════════════════════════════════
#  VIDEO CHATBOT
# ═════════════════════════════════════════════════════════

class VideoChatIndex:
    def __init__(self, video_id):
        self.video_id=video_id
        self.index_dir=Path(INDEX_DIR)/video_id; self.index_dir.mkdir(parents=True,exist_ok=True)
        self.chunks=[]; self.embeddings=None; self._loaded=False

    def build(self, unified, stage1):
        self.chunks=[]; texts=[]
        for seg in unified["unified_segments"]:
            self.chunks.append({"start":seg["start"],"end":seg["end"],"text":seg["text"],
                                 "visual_caps":seg["visual_caps"],"combined_text":seg["combined_text"]})
            texts.append(seg["combined_text"])
        if not texts: return
        em=get_embed(); self.embeddings=em.encode(texts,show_progress_bar=False)
        with open(self.index_dir/"index.json","w") as f:
            json.dump({"video_id":self.video_id,"chunks":self.chunks,
                       "embeddings_shape":list(self.embeddings.shape)},f)
        np.save(self.index_dir/"embeddings.npy",self.embeddings); self._loaded=True

    def load(self):
        jp=self.index_dir/"index.json"; ep=self.index_dir/"embeddings.npy"
        if not jp.exists() or not ep.exists(): return False
        with open(jp) as f: data=json.load(f)
        self.chunks=data["chunks"]; self.embeddings=np.load(ep); self._loaded=True; return True

    def search(self, query, top_k=3):
        if not self._loaded or self.embeddings is None: return []
        em=get_embed(); q_emb=em.encode([query])
        sims=cosine_similarity(q_emb,self.embeddings)[0]; top_idx=np.argsort(sims)[::-1][:top_k]
        results=[]
        for i in top_idx:
            c=self.chunks[i].copy(); c["relevance_score"]=round(float(sims[i]),4)
            c["timestamp_str"]=hhmmss(c["start"]); results.append(c)
        return results

    def chat(self, query, top_k=3, use_vlm=False):
        hits=self.search(query,top_k=top_k)
        if not hits: return {"answer":"No relevant content found.","sources":[]}
        context="\n".join(f"[{h['timestamp_str']}] {h['combined_text']}" for h in hits)
        if use_vlm:
            try:
                proc,model=get_qwen()
                if hasattr(proc,"apply_chat_template"):
                    vlm_prompt=f"Based on this video content:\n{context}\n\nAnswer: {query}\nBe concise."
                    msgs=[{"role":"user","content":[{"type":"text","text":vlm_prompt}]}]
                    text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
                    inputs=proc(text=text,return_tensors="pt").to(GPU_QWEN)
                    with torch.no_grad(): out=model.generate(**inputs,max_new_tokens=100)
                    raw=proc.decode(out[0],skip_special_tokens=True).strip()
                    # Strip prompt leakthrough
                    for marker in ["assistant", "ASSISTANT"]:
                        if marker in raw:
                            raw = raw.split(marker)[-1].strip().lstrip(":").strip()
                    answer=raw
                else: answer=f"[{hits[0]['timestamp_str']}] {hits[0]['text']}"
            except Exception: answer=f"[{hits[0]['timestamp_str']}] {hits[0]['text']}"
        else: answer=f"Most relevant: {hits[0]['timestamp_str']}\n{hits[0]['combined_text']}"
        return {"answer":answer,"sources":hits,"context_used":context[:500]}

_chat_indexes: Dict[str, VideoChatIndex] = {}

# ═════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════

BRAILLE_MAP={"a":"⠁","b":"⠃","c":"⠉","d":"⠙","e":"⠑","f":"⠋","g":"⠛","h":"⠓",
 "i":"⠊","j":"⠚","k":"⠅","l":"⠇","m":"⠍","n":"⠝","o":"⠕","p":"⠏",
 "q":"⠟","r":"⠗","s":"⠎","t":"⠞","u":"⠥","v":"⠧","w":"⠺","x":"⠭",
 "y":"⠽","z":"⠵"," ":" ",",":"⠂",".":"⠲","?":"⠦","!":"⠖","-":"⠤","\n":"\n"}
DIGS={"0":"⠚","1":"⠁","2":"⠃","3":"⠉","4":"⠙","5":"⠑","6":"⠋","7":"⠛","8":"⠓","9":"⠊"}
SUPPORTED_LANGUAGES={"Hindi":"hi","French":"fr","German":"de","Spanish":"es","Korean":"ko","Japanese":"ja"}

def hhmmss(t):
    t=max(0.0,float(t)); h=int(t//3600); t-=h*3600; m=int(t//60); s=int(t%60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

def _to_braille(text):
    return "".join("⠼"+DIGS[c] if c.isdigit() else BRAILLE_MAP.get(c.lower(),c) for c in text)

def text_to_asl(text):
    return [{"char":c.lower(),"path":f"/static/asl_letters/{c.lower()}.mp4"} if c.isalpha()
            else {"char":"space","path":"/static/asl_letters/space.mp4"} if c==" "
            else {"char":c,"path":f"/static/asl_letters/{c}.mp4"} if c.isdigit() else None
            for c in text if c.isalpha() or c==" " or c.isdigit()]

def generate_summary(t, n=4):
    if not t.strip(): return "No summary."
    s=[x.strip() for x in re.split(r"(?<=[.!?])\s+",t) if len(x.strip())>15]
    return " ".join(s[:n]) if s else t[:300]

def extract_key_concepts(t, n=6):
    if not t.strip(): return ["No key concepts."]
    s=[x.strip() for x in re.split(r"(?<=[.!?])\s+",t) if len(x.strip())>20]
    if not s: return [t[:120]]
    if len(s)<=n: return s
    idx=[int(round(i*(len(s)-1)/(n-1))) for i in range(n)]; seen,out=set(),[]
    for i in idx:
        if i not in seen: seen.add(i); out.append(s[i])
    return out

def extract_youtube_id(url):
    for p in [r"(?:v=)([a-zA-Z0-9_-]{11})",r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
              r"(?:embed/)([a-zA-Z0-9_-]{11})",r"(?:shorts/)([a-zA-Z0-9_-]{11})"]:
        m=re.search(p,url)
        if m: return m.group(1)
    return ""

def download_youtube(url, d):
    prefix=uuid.uuid4().hex[:8]
    opts={"format":"bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
          "outtmpl":os.path.join(d,f"{prefix}_%(id)s.%(ext)s"),
          "merge_output_format":"mp4","quiet":True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info=ydl.extract_info(url,download=True); fp=ydl.prepare_filename(info)
    return fp if os.path.exists(fp) else re.sub(r"\.\w+$",".mp4",fp)

def extract_frames(video_path, step=FRAME_STEP_SECONDS, max_f=MAX_FRAMES):
    frames=[]
    with VideoFileClip(video_path) as clip:
        dur=float(clip.duration or 0.0)
        if dur<=0: return frames,0.0
        ts=np.arange(0.0,dur,step)
        if len(ts)>max_f: ts=ts[np.linspace(0,len(ts)-1,max_f).astype(int)]
        for t in ts:
            arr=clip.get_frame(float(min(t,max(0.0,dur-1e-3))))
            img=Image.fromarray(arr).convert("RGB"); img.thumbnail((512,512))
            frames.append((float(t),img))
    return frames,dur

def build_enriched_transcript(unified, stage4):
    lines=[]
    for seg in unified["unified_segments"]:
        s,e,txt=seg["start"],seg["end"],seg["text"]
        tag=(f"<a class='timestamp' href='#' onclick='seekTo({s}); return false;' "
             f"title='Jump to {hhmmss(s)}'>[{hhmmss(s)}–{hhmmss(e)}]</a>")
        vcap=""
        if seg["visual_caps"]:
            vcap=f"  <span class='visual-tag'>[Visual: {'|'.join(seg['visual_caps'][:2])}]</span>"
        lines.append(f"{tag} {txt}{vcap}")
    return "<br>".join(lines)

# ═════════════════════════════════════════════════════════
#  KEYFRAME SAVING FOR FORENSIC RECONSTRUCTION CARD
# ═════════════════════════════════════════════════════════

def save_keyframes(frames_ts, video_hash):
    """Save keyframe thumbnails to uploads dir, return list of {t, path, caption}."""
    keyframes=[]
    for idx,(t,img) in enumerate(frames_ts):
        if idx % KEYFRAME_INTERVAL != 0: continue
        fn  = f"{video_hash}_kf_{idx:04d}.jpg"
        fp  = os.path.join(UPLOAD_FOLDER, fn)
        thumb = img.copy(); thumb.thumbnail((400,400))
        thumb.save(fp, "JPEG", quality=85)
        keyframes.append({"t":round(float(t),2),"path":f"/uploads/{fn}","idx":idx})
    return keyframes

# ═════════════════════════════════════════════════════════
#  MODE-SPECIFIC PIPELINES
# ═════════════════════════════════════════════════════════

class AsyncResult:
    def __init__(self): self._l=threading.Lock(); self._d={}; self._e={}
    def set(self,k,v):
        with self._l: self._d[k]=v
    def get(self,k,d=None):
        with self._l: return self._d.get(k,d)
    def error(self,k,e):
        with self._l: self._e[k]=str(e); traceback.print_exc()
    def errors(self):
        with self._l: return dict(self._e)

# ─── MODE 1: Video to Text ───────────────────────────────

def process_video_text(video_path, reference_text, progress_cb=None):
    """Lightweight: Whisper + CLIP captions + chatbot only."""
    def cb(step,msg):
        if progress_cb: progress_cb(step,msg)

    audio_path=os.path.splitext(video_path)[0]+".wav"
    try:
        cb(1,"Extracting audio and frames...")
        with VideoFileClip(video_path) as video:
            if video.audio is None: 
                print("[Warning] No audio track — skipping audio extraction.")
                open(audio_path, "w").close()
            video.audio.write_audiofile(audio_path,logger=None)
        frames_ts, duration = extract_frames(video_path)
        video_hash = uuid.uuid4().hex[:12]

        cb(2,"Running CLIP + Qwen captions...")
        stage1 = run_stage1_video_to_text(frames_ts, video_hash, cb, forensic_mode=False)

        cb(3,"Running Whisper transcription...")
        stage2 = run_stage2_audio_to_text(audio_path, cb)

        cb(4,"Building unified representation...")
        unified = build_unified_representation(stage1, stage2, duration)

        cb(5,"Building chatbot index...")
        chat_idx = VideoChatIndex(video_hash)
        chat_idx.build(unified, stage1)
        _chat_indexes[video_hash] = chat_idx

        cb(6,"Assembling results...")
        transcript = stage2["transcript"]
        enriched   = build_enriched_transcript(unified, {"fusion":{}})

        # Minimal ref score
        ref_score=0.0
        if reference_text.strip() and transcript.strip():
            em=get_embed()
            ref_score=float(cosine_similarity(em.encode([transcript[:512]]),em.encode([reference_text]))[0][0])

        return {
            "mode":              "video_to_text",
            "enriched_transcript": enriched,
            "transcript":        transcript,
            "score":             round(ref_score*100.0,2),
            "braille":           _to_braille(transcript),
            "summary":           generate_summary(transcript),
            "key_concepts":      extract_key_concepts(transcript),
            "audio_filename":    os.path.basename(audio_path),
            "asl_entries":       text_to_asl(transcript),
            "video_duration":    duration,
            "language":          stage2["language"],
            "unified_segments":  unified["unified_segments"][:50],
            "scene_labels":      stage1.get("scene_labels", [])[:20],
            "n_keyframes_analyzed": stage1.get("n_keyframes", 0),
            "video_hash":        video_hash,
            "chatbot_ready":     True,
        }
    except Exception:
        if os.path.exists(audio_path):
            try: os.remove(audio_path)
            except: pass
        raise

# ─── MODE 2: Forensic Analysis ───────────────────────────

def process_video_forensic(video_path, reference_text, progress_cb=None):
    """Full pipeline — all GPUs, all modules."""
    def cb(step,msg):
        if progress_cb: progress_cb(step,msg)

    audio_path=os.path.splitext(video_path)[0]+".wav"
    try:
        cb(1,"Extracting audio and frames...")
        with VideoFileClip(video_path) as video:
            if video.audio is None: 
                print("[Warning] No audio track — skipping audio extraction.")
                open(audio_path, "w").close()
            video.audio.write_audiofile(audio_path,logger=None)
        frames_ts, duration = extract_frames(video_path)
        video_hash = uuid.uuid4().hex[:12]

        cb(2,"Parallel visual + audio analysis...")
        ar = AsyncResult(); ar.set("duration", duration)
        def run_s1(): ar.set("stage1", run_stage1_video_to_text(frames_ts, video_hash, cb, forensic_mode=True))
        def run_s2(): ar.set("stage2", run_stage2_audio_to_text(audio_path, cb))
        t1=threading.Thread(target=run_s1,daemon=True)
        t2=threading.Thread(target=run_s2,daemon=True)
        t1.start(); t2.start(); t1.join(); t2.join()

        stage1 = ar.get("stage1",{})
        stage2 = ar.get("stage2",{"transcript":"","segments":[],"language":"","word_count":0})

        cb(3,"Analyzing audio...")
        audio_activity = analyze_audio_activity(audio_path, stage2["segments"])
        stage2["_audio_features"] = audio_activity

        cb(3,"Building unified representation...")
        unified = build_unified_representation(stage1, stage2, duration)
        stage3  = run_stage3_text_to_video(frames_ts, stage1, stage2, unified, cb)

        cb(4,"Deepfake detection (CNN + FFT)...")
        print(f"[Deepfake] Starting - frames_ts has {len(frames_ts)} frames")
        df_frame_results=[]; mean_fake_score=0.0
        try:
            kf_frames=[(t,img) for idx,(t,img) in enumerate(frames_ts) if idx%KEYFRAME_INTERVAL==0]
            kf_frames=kf_frames[:MAX_DEEPFAKE_KF]
            for t,img in kf_frames:
                r=detect_deepfake_frame(img); r["t"]=round(float(t),2); df_frame_results.append(r)
            if df_frame_results:
                mean_fake_score=round(float(np.mean([r["fake_score"] for r in df_frame_results])),4)

            torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            print(f"[Deepfake] ERROR: {e}")
            traceback.print_exc()

        deepfake_results={"frame_results":df_frame_results,"mean_fake_score":mean_fake_score,
                          "max_fake_score":round(float(max((r["fake_score"] for r in df_frame_results),default=0.0)),4),
                          "n_frames":len(df_frame_results),
                          "fake_frame_ratio":round(float(np.mean([r["fake_score"]>0.5 for r in df_frame_results])),4) if df_frame_results else 0.0,
                          "verdict":"FAKE" if mean_fake_score>0.55 else "REAL"}

        cb(5,"Temporal inconsistency detection...")
        timestamps=[float(t) for t,_ in frames_ts]
        temporal_results=detect_temporal_inconsistencies(stage1.get("clip_embs", []),timestamps)

        cb(5,"Scene detection...")
        scene_results=detect_scenes(frames_ts,stage1.get("clip_embs", []))

        cb(5,"Emotion detection (GPU 3)...")
        emotion_results=analyze_emotion_timeline(frames_ts,keyframe_interval=KEYFRAME_INTERVAL)

        cb(5,"Physiological signal analysis (rPPG)...")
        fps_est=len(frames_ts)/max(duration,1.0)
        physiological=estimate_rppg_pulse(frames_ts,fps=fps_est)

        cb(5,"CLIP multimodal alignment...")
        mm_alignment=compute_multimodal_alignment_clip(frames_ts,stage2["segments"])

        stage4=run_stage4_fusion(stage1,stage2,stage3,reference_text,
                                  temporal_results,physiological,audio_activity,cb,deepfake_results=deepfake_results)

        cb(6,"Building analytical report...")
        analytical_report=build_analytical_report(
            stage1,stage2,stage3,stage4,deepfake_results,temporal_results,
            scene_results,emotion_results,audio_activity,mm_alignment,
            physiological,unified,duration)

        cb(7,"Building chatbot index...")
        chat_idx=VideoChatIndex(video_hash); chat_idx.build(unified,stage1)
        _chat_indexes[video_hash]=chat_idx

        cb(7,"Saving keyframes for reconstruction...")
        keyframes=save_keyframes(frames_ts,video_hash)

        # Attach captions to keyframes for reconstruction card
        caption_map={c["t"]:c["caption"] for c in stage1.get("captions", []) if c["keyframe"]}
        for kf in keyframes:
            kf["caption"]=caption_map.get(kf["t"],"")
            # DRE score per keyframe
            dre_map={d["t"]:d["ood_score"] for d in stage3["dre_scores"]}
            nearest_t=min(dre_map.keys(), key=lambda x: abs(x-kf["t"])) if dre_map else None
            kf["ood_score"]=dre_map.get(nearest_t,0.0) if nearest_t else 0.0
            # Deepfake score per keyframe
            df_map={r["t"]:r["fake_score"] for r in df_frame_results}
            nearest_df=min(df_map.keys(), key=lambda x: abs(x-kf["t"])) if df_map else None
            kf["fake_score"]=df_map.get(nearest_df,0.0) if nearest_df else 0.0

        cb(8,"Assembling final report...")
        fusion=stage4["fusion"]
        transcript=stage2["transcript"]
        enriched=build_enriched_transcript(unified,stage4)
        forensic_segments=[{"start":s["start"],"end":s["end"],
                             "text":s["text"][:80]+("…" if len(s["text"])>80 else ""),
                             "label":fusion.get("label","unknown"),
                             "ai_score":fusion.get("ai_score",0.0)}
                            for s in unified["unified_segments"]]
        vlm_backend="qwen2vl_2b"
        try:
            r=get_qwen()
            if hasattr(r,"__len__") and len(r)==2 and not hasattr(r[0],"apply_chat_template"):
                vlm_backend="blip_fallback"
        except: pass

        return {
            "mode":                 "forensic",
            "enriched_transcript":  enriched,
            "transcript":           transcript,
            "score":                round(stage4["ref_score"]*100.0,2),
            "braille":              _to_braille(transcript),
            "summary":              generate_summary(transcript),
            "key_concepts":         extract_key_concepts(transcript),
            "audio_filename":       os.path.basename(audio_path),
            "asl_entries":          text_to_asl(transcript),
            "video_duration":       duration,
            "language":             stage2["language"],
            "unified_segments":     unified["unified_segments"][:50],
            "alignment":            stage3["alignment"],
            "scene_labels":         stage1.get("scene_labels", [])[:20],
            "n_keyframes_analyzed": stage1.get("n_keyframes", 0),
            "video_forensic_label": fusion.get("label","unknown"),
            "video_ai_score":       round(fusion.get("ai_score",0.0),4),
            "forensic_segments":    forensic_segments,
            "audio_forensics":      {"av_sync":stage3["av_sync"],
                                     "clone_score":audio_activity.get("clone_score",0.3),
                                     "synthetic_score":round(float(np.clip(
                                         0.4*stage3["dre_mean"]+0.3*stage1.get("ai_frame_ratio", 0)+
                                         0.3*(1-stage3["av_sync"]["sync_score"]),0,1)),4),
                                     "audio_activity":audio_activity},
            "temporal_consistency": stage3["temporal_consistency"],
            "dre_summary":          {"mean_ood":stage3["dre_mean"],"max_ood":stage3["dre_max"],
                                     "n_frames":len(stage3["dre_scores"])},
            "moe_fusion":           fusion,
            "vlm_backend":          vlm_backend,
            "video_hash":           video_hash,
            "chatbot_ready":        True,
            "lane_errors":          ar.errors(),
            "deepfake_analysis":    deepfake_results,
            "temporal_analysis":    temporal_results,
            "scene_detection":      scene_results,
            "emotion_analysis":     emotion_results,
            "multimodal_alignment": mm_alignment,
            "physiological":        physiological,
            "analytical_report":    analytical_report,
            "keyframes":            keyframes,   # for reconstruction card
        }
    except Exception:
        if os.path.exists(audio_path):
            try: os.remove(audio_path)
            except: pass
        raise

# ─── MODE 3: Live (file upload) — stripped pipeline ──────

def process_video_live(video_path, progress_cb=None):
    """Stripped: CLIP + DeepfakeCNN + rPPG + Emotion only. No Whisper/Qwen."""
    def cb(step,msg):
        if progress_cb: progress_cb(step,msg)

    try:
        cb(1,"Extracting frames (live mode — no audio transcription)...")
        frames_ts, duration = extract_frames(video_path, step=2.0, max_f=60)
        video_hash = uuid.uuid4().hex[:12]
        timestamps = [float(t) for t,_ in frames_ts]

        cb(2,"CLIP embeddings (live mode)...")
        clip_embs=[]
        for idx,(t,img) in enumerate(frames_ts):
            emb=clip_embed_frame(img); clip_embs.append(emb)

        cb(3,"Deepfake detection (live mode)...")
        df_frame_results=[]
        for idx,(t,img) in enumerate(frames_ts):
            if idx%3!=0: continue  # every 3rd frame in live
            r=detect_deepfake_frame(img); r["t"]=round(float(t),2); df_frame_results.append(r)
        torch.cuda.empty_cache()

        cb(4,"Emotion timeline (live mode)...")
        emotion_results=analyze_emotion_timeline(frames_ts,keyframe_interval=3)

        cb(4,"rPPG physiological (live mode)...")
        fps_est=len(frames_ts)/max(duration,1.0)
        physiological=estimate_rppg_pulse(frames_ts,fps=fps_est)

        cb(5,"Temporal consistency...")
        temporal_results=detect_temporal_inconsistencies(clip_embs,timestamps)

        cb(5,"Scene detection (live mode)...")
        scene_results=detect_scenes(frames_ts,clip_embs)

        mean_fake=round(float(np.mean([r["fake_score"] for r in df_frame_results])),4) if df_frame_results else 0.0
        temporal_risk=min(temporal_results.get("mean_anomaly",0.0)*2,1.0)
        liveness=physiological.get("liveness_score",0.5)
        ai_risk=round(float(np.clip(0.3*temporal_risk+0.7*(1.0-liveness),0,1)),4)

        cb(6,"Done.")
        return {
            "mode":              "live",
            "video_hash":        video_hash,
            "video_duration":    duration,
            "ai_risk_score":     ai_risk,
            "deepfake_analysis": {"frame_results":df_frame_results,"mean_fake_score":mean_fake,
                                  "verdict":"FAKE" if mean_fake>0.65 else "REAL"},
            "temporal_analysis": temporal_results,
            "emotion_analysis":  emotion_results,
            "physiological":     physiological,
            "scene_detection":   scene_results,
            "rolling_scores":    [{"t":r["t"],"fake_score":r["fake_score"]} for r in df_frame_results],
            "note":              "Live mode: no audio transcription. Upload in Forensic mode for full analysis.",
        }
    except Exception:
        raise

# ═════════════════════════════════════════════════════════
#  WEBCAM LIVE SESSION — frame-by-frame processing
# ═════════════════════════════════════════════════════════

def process_webcam_frame(session_id: str, b64_frame: str) -> dict:
    """
    Process a single webcam frame (base64 JPEG).
    Runs: DeepfakeCNN + EmotionClassifier + rPPG signal accumulation.
    Returns rolling scores for live dashboard update.
    """
    # Decode frame
    try:
        img_bytes = base64.b64decode(b64_frame)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        img.thumbnail((512,512))
    except Exception as e:
        return {"error":f"Frame decode failed: {e}"}

    with live_sessions_lock:
        if session_id not in live_sessions:
            return {"error":"Session not found"}
        sess = live_sessions[session_id]

    t = time.time() - sess["start_time"]

    # Deepfake detection
    try:
        df_result = detect_deepfake_frame(img)
        fake_score = df_result["fake_score"]
    except Exception as e:
        fake_score = 0.0
        df_result  = {"fake_score":0.0,"verdict":"UNKNOWN","error":str(e)}

    # Emotion detection
    try:
        em_result = detect_emotion_frame(img, t)
        emotion   = em_result["emotion"]
        em_conf   = em_result["confidence"]
    except Exception:
        emotion="unknown"; em_conf=0.0; em_result={}

    # rPPG — accumulate green channel signal
    try:
        faces = extract_faces(img)
        if faces:
            face = faces[0]; arr=np.array(face.resize((64,64))).astype(np.float32)
            forehead=arr[:int(0.30*arr.shape[0]),:,:]
            g_mean=float(np.mean(forehead[:,:,1]))
        else:
            g_mean=None
    except Exception:
        g_mean=None

    # CLIP embed for temporal consistency
    try:
        emb = clip_embed_frame(img)
    except Exception:
        emb = None

    # Update session state
    with live_sessions_lock:
        sess = live_sessions[session_id]
        sess["frames"].append({"t":round(t,2),"fake_score":fake_score,"emotion":emotion,
                               "em_conf":em_conf,"g_mean":g_mean})
        if emb is not None: sess["clip_embs"].append(emb)
        sess["frame_count"]+=1

        # Rolling AI risk (last 10 frames)
        recent = sess["frames"][-10:]
        roll_fake = float(np.mean([f["fake_score"] for f in recent]))

        # Quick rPPG on accumulated signal
        g_vals=[f["g_mean"] for f in sess["frames"] if f["g_mean"] is not None]
        pulse_bpm=0.0; liveness=0.5
        if len(g_vals)>=16:
            sig=np.array(g_vals,dtype=np.float64)-np.mean(g_vals)
            fps_est=sess["frame_count"]/max(t,1.0)
            freqs=np.fft.rfftfreq(len(sig),d=1.0/fps_est)
            fft_mag=np.abs(np.fft.rfft(sig))
            band=(freqs>=0.7)&(freqs<=4.0)
            if band.any():
                band_mag=fft_mag*band; pk=int(np.argmax(band_mag))
                pulse_bpm=round(float(freqs[pk])*60.0,1)
                conf=float(np.clip(band_mag[pk]/(np.sum(fft_mag)+1e-9)*10,0,1))
                liveness=round(float(np.clip(conf*(1.0 if 42<=pulse_bpm<=150 else 0.2),0,1)),4)

        # Temporal consistency (last 5 embeds)
        tc=1.0
        if len(sess["clip_embs"])>=2:
            embs=sess["clip_embs"][-5:]
            sims=[float(cosine_similarity(embs[i][None],embs[i+1][None])[0,0])
                  for i in range(len(embs)-1)]
            tc=round(float(np.mean(sims)),4)

        ai_risk=round(float(np.clip(0.5*(1.0-tc)+0.5*(1.0-liveness),0,1)),4)
        sess["last_ai_risk"]=ai_risk
        verdict="FAKE" if (ai_risk>0.65 and sess["frame_count"]>=15) else ("REAL" if sess["frame_count"]>=15 else "COLLECTING...")

    return {"t":round(t,2),"frame_count":sess["frame_count"],
            "fake_score":round(fake_score,4),"rolling_fake":round(roll_fake,4),
            "emotion":emotion,"em_confidence":round(em_conf,4),
            "pulse_bpm":pulse_bpm,"liveness_score":liveness,
            "temporal_consistency":tc,"ai_risk_score":ai_risk,
            "verdict":verdict,"df_detail":df_result}

# ═════════════════════════════════════════════════════════
#  BACKGROUND JOB
# ═════════════════════════════════════════════════════════

def run_job(jid, video_path, reference_text, unique_name, mode):
    def cb(step,msg): set_job(jid,progress=msg,step=step)
    try:
        set_job(jid,status="processing",progress="Starting...",step=0,mode=mode)
        if mode=="video_to_text":
            r = process_video_text(video_path, reference_text, progress_cb=cb)
            result_route = f"/results/text/{jid}"
        elif mode=="forensic":
            r = process_video_forensic(video_path, reference_text, progress_cb=cb)
            result_route = f"/results/forensic/{jid}"
        elif mode=="live":
            r = process_video_live(video_path, progress_cb=cb)
            result_route = f"/results/live/{jid}"
        else:
            raise ValueError(f"Unknown mode: {mode}")

        payload={**r,"audio_url":f"/uploads/{r.get('audio_filename','')}",
                 "video_url":f"/uploads/{unique_name}"}
        set_job(jid,status="done",progress="Complete! ✓",step=8,
                result=payload,video_hash=r.get("video_hash",""),
                result_route=result_route,mode=mode)
    except Exception as e:
        traceback.print_exc()
        set_job(jid,status="error",progress="Failed.",error=str(e))

# ═════════════════════════════════════════════════════════
#  FLASK ROUTES
# ═════════════════════════════════════════════════════════

@app.after_request
def no_cache(r):
    r.headers["Cache-Control"]="no-cache, no-store, must-revalidate"
    r.headers["Pragma"]="no-cache"
    r.headers["Expires"]="0"
    return r

@app.route("/")
def home(): return render_template("index.html")

@app.route("/upload_file", methods=["POST"])
def upload_file():
    if "file" not in request.files: return jsonify({"error":"No file"}),400
    file=request.files["file"]
    ref=(request.form.get("reference") or "").strip()
    mode=(request.form.get("mode") or "forensic").strip()
    if mode not in VALID_MODES: mode="forensic"
    if not file or file.filename=="": return jsonify({"error":"No file selected"}),400
    if not allowed_file(file.filename): return jsonify({"error":"Unsupported type"}),400
    fn=secure_filename(file.filename); un=f"{uuid.uuid4().hex[:8]}_{fn}"
    fp=os.path.join(app.config["UPLOAD_FOLDER"],un); file.save(fp)
    jid=uuid.uuid4().hex
    with jobs_lock: jobs[jid]={"status":"pending","progress":"Queued...","step":0,
                                "result":None,"error":None,"video_hash":None,"mode":mode}
    threading.Thread(target=run_job,args=(jid,fp,ref,un,mode),daemon=True).start()
    return jsonify({"job_id":jid,"mode":mode})

@app.route("/upload", methods=["POST"])
def upload_youtube():
    yt_url=(request.form.get("youtube_url") or "").strip()
    ref=(request.form.get("reference") or "").strip()
    mode=(request.form.get("mode") or "forensic").strip()
    if mode not in VALID_MODES: mode="forensic"
    if not yt_url: return "URL required",400
    vid=extract_youtube_id(yt_url)
    if not vid: return "Invalid URL",400
    jid=uuid.uuid4().hex
    with jobs_lock: jobs[jid]={"status":"processing","progress":"Downloading video...","step":0,
                                "result":None,"error":None,"video_hash":None,"mode":mode}
    def download_and_run():
        try:
            video_path=download_youtube(yt_url,app.config["UPLOAD_FOLDER"])
            un=os.path.basename(video_path)
            run_job(jid,video_path,ref,un,mode)
        except Exception as e:
            with jobs_lock: 
                jobs[jid]["status"]="error"
                jobs[jid]["error"]=str(e)
                jobs[jid]["progress"]="Download failed"
    threading.Thread(target=download_and_run,daemon=True).start()
    return jsonify({"job_id":jid,"mode":mode})

@app.route("/status/<jid>")
def status(jid):
    with jobs_lock: j=jobs.get(jid)
    if not j: return jsonify({"error":"Not found"}),404
    return jsonify({k:v for k,v in j.items() if k!="result"})

# ─── Result routes ────────────────────────────────────────

@app.route("/results/text/<jid>")
def results_text(jid):
    with jobs_lock: j=jobs.get(jid)
    if not j or j["status"]!="done": return "Results not ready",404
    r=j["result"]
    session.update({"transcript":r["transcript"],"braille":r["braille"],
                    "translations":{},"video_hash":r.get("video_hash","")})
    return render_template("results_text.html",
        enriched_transcript=r["enriched_transcript"],
        transcript=r["transcript"],score=r["score"],braille=r["braille"],
        summary=r["summary"],key_concepts=r["key_concepts"],
        audio_url=r["audio_url"],video_url=r["video_url"],
        asl_entries=r["asl_entries"],video_duration=r.get("video_duration",0),
        supported_languages=list(SUPPORTED_LANGUAGES.keys()),
        language=r.get("language",""),
        scene_labels=r.get("scene_labels",[]),
        n_keyframes_analyzed=r.get("n_keyframes_analyzed",0),
        chatbot_ready=r.get("chatbot_ready",False),
        video_hash=r.get("video_hash",""))

@app.route("/results/forensic/<jid>")
def results_forensic(jid):
    with jobs_lock: j=jobs.get(jid)
    if not j or j["status"]!="done": return "Results not ready",404
    r=j["result"]
    session.update({"transcript":r["transcript"],"braille":r["braille"],
                    "translations":{},"video_hash":r.get("video_hash","")})
    return render_template("results_forensic.html",
        enriched_transcript=r["enriched_transcript"],
        transcript=r["transcript"],score=r["score"],
        audio_url=r["audio_url"],video_url=r["video_url"],
        video_duration=r.get("video_duration",0),
        language=r.get("language",""),
        video_forensic_label=r.get("video_forensic_label","unknown"),
        composite_risk_score=r.get("composite_risk_score",0.0),
        video_ai_score=r.get("video_ai_score",0.0),
        forensic_segments=r.get("forensic_segments",[]),
        audio_forensics=r.get("audio_forensics",{}),
        temporal_consistency=r.get("temporal_consistency",0.0),
        vlm_backend=r.get("vlm_backend","unknown"),
        dre_summary=r.get("dre_summary",{}),
        moe_fusion=r.get("moe_fusion",{}),
        alignment=r.get("alignment",{}),
        scene_labels=r.get("scene_labels",[]),
        chatbot_ready=r.get("chatbot_ready",False),
        video_hash=r.get("video_hash",""),
        deepfake_analysis=r.get("deepfake_analysis",{}),
        temporal_analysis=r.get("temporal_analysis",{}),
        scene_detection=r.get("scene_detection",{}),
        emotion_analysis=r.get("emotion_analysis",{}),
        multimodal_alignment=r.get("multimodal_alignment",{}),
        physiological=r.get("physiological",{}),
        analytical_report=r.get("analytical_report",{}),
        keyframes=r.get("keyframes",[]),
        n_keyframes_analyzed=r.get("n_keyframes_analyzed",0),
        supported_languages=list(SUPPORTED_LANGUAGES.keys()),
        braille=r.get("braille",""),
        asl_entries=r.get("asl_entries",[]),
        summary=r.get("summary",""),
        key_concepts=r.get("key_concepts",[]))

@app.route("/results/live/<jid>")
def results_live(jid):
    with jobs_lock: j=jobs.get(jid)
    if not j or j["status"]!="done": return "Results not ready",404
    r=j["result"]
    return render_template("results_live.html",
        video_url=r["video_url"],video_duration=r.get("video_duration",0),
        ai_risk_score=r.get("ai_risk_score",0.0),
        deepfake_analysis=r.get("deepfake_analysis",{}),
        temporal_analysis=r.get("temporal_analysis",{}),
        emotion_analysis=r.get("emotion_analysis",{}),
        physiological=r.get("physiological",{}),
        scene_detection=r.get("scene_detection",{}),
        rolling_scores=r.get("rolling_scores",[]),
        note=r.get("note",""),video_hash=r.get("video_hash",""))

# ─── Webcam live session routes ───────────────────────────

@app.route("/webcam/start", methods=["POST"])
def webcam_start():
    session_id = uuid.uuid4().hex
    with live_sessions_lock:
        live_sessions[session_id] = {
            "start_time":   time.time(),
            "frames":       [],
            "clip_embs":    [],
            "frame_count":  0,
            "last_ai_risk": 0.0,
            "active":       True,
        }
    return jsonify({"session_id": session_id})

@app.route("/webcam/frame", methods=["POST"])
def webcam_frame():
    data = request.get_json(force=True, silent=True) or {}
    session_id = data.get("session_id","")
    b64_frame  = data.get("frame","")
    if not session_id or not b64_frame:
        return jsonify({"error":"session_id and frame required"}),400
    result = process_webcam_frame(session_id, b64_frame)
    return jsonify(result)

@app.route("/webcam/stop/<session_id>", methods=["POST"])
def webcam_stop(session_id):
    with live_sessions_lock:
        if session_id not in live_sessions:
            return jsonify({"error":"Session not found"}),404
        sess = live_sessions.pop(session_id)
    frames=sess["frames"]
    if not frames:
        return jsonify({"summary":{},"frames":[]})
    fake_scores=[f["fake_score"] for f in frames]
    emotions=[f["emotion"] for f in frames]
    emotion_counts=Counter(emotions)
    g_vals=[f["g_mean"] for f in frames if f["g_mean"] is not None]
    pulse_bpm=0.0
    if len(g_vals)>=16:
        sig=np.array(g_vals,dtype=np.float64)-np.mean(g_vals)
        fps_est=len(frames)/max(time.time()-sess.get("start_time",time.time()-1),1.0)
        freqs=np.fft.rfftfreq(len(sig),d=1.0/max(fps_est,1.0))
        fft_mag=np.abs(np.fft.rfft(sig)); band=(freqs>=0.7)&(freqs<=4.0)
        if band.any():
            band_mag=fft_mag*band; pk=int(np.argmax(band_mag)); pulse_bpm=round(float(freqs[pk])*60.0,1)
    return jsonify({
        "session_id":       session_id,
        "total_frames":     len(frames),
        "duration_sec":     round(frames[-1]["t"] if frames else 0.0,2),
        "mean_fake_score":  round(float(np.mean(fake_scores)),4),
        "max_fake_score":   round(float(np.max(fake_scores)),4),
        "verdict":          "FAKE" if sess.get("last_ai_risk",0)>0.65 else "REAL",
        "dominant_emotion": emotion_counts.most_common(1)[0][0] if emotion_counts else "unknown",
        "emotion_counts":   dict(emotion_counts),
        "estimated_pulse_bpm": pulse_bpm,
        "overall_ai_risk":  sess.get("last_ai_risk",0.0),
        "frames":           frames[:200],
    })

# ─── JSON API routes ──────────────────────────────────────

@app.route("/forensic/<jid>")
def forensic(jid):
    with jobs_lock: j=jobs.get(jid)
    if not j: return jsonify({"error":"Not found"}),404
    if j["status"]!="done": return jsonify({"status":j["status"],"progress":j["progress"]}),202
    r=j["result"]
    return jsonify({k:r.get(k) for k in ["video_forensic_label","video_ai_score","forensic_segments",
        "audio_forensics","temporal_consistency","dre_summary","moe_fusion","alignment","vlm_backend",
        "deepfake_analysis","temporal_analysis","scene_detection","emotion_analysis",
        "multimodal_alignment","physiological","analytical_report","keyframes"]})

@app.route("/analysis/deepfake/<jid>")
def analysis_deepfake(jid):
    with jobs_lock: j=jobs.get(jid)
    if not j or j["status"]!="done": return jsonify({"error":"Not ready"}),404
    return jsonify(j["result"].get("deepfake_analysis",{}))

@app.route("/analysis/temporal/<jid>")
def analysis_temporal(jid):
    with jobs_lock: j=jobs.get(jid)
    if not j or j["status"]!="done": return jsonify({"error":"Not ready"}),404
    return jsonify(j["result"].get("temporal_analysis",{}))

@app.route("/analysis/scenes/<jid>")
def analysis_scenes(jid):
    with jobs_lock: j=jobs.get(jid)
    if not j or j["status"]!="done": return jsonify({"error":"Not ready"}),404
    return jsonify(j["result"].get("scene_detection",{}))

@app.route("/analysis/emotions/<jid>")
def analysis_emotions(jid):
    with jobs_lock: j=jobs.get(jid)
    if not j or j["status"]!="done": return jsonify({"error":"Not ready"}),404
    return jsonify(j["result"].get("emotion_analysis",{}))

@app.route("/analysis/alignment/<jid>")
def analysis_alignment(jid):
    with jobs_lock: j=jobs.get(jid)
    if not j or j["status"]!="done": return jsonify({"error":"Not ready"}),404
    return jsonify(j["result"].get("multimodal_alignment",{}))

@app.route("/analysis/physiological/<jid>")
def analysis_physiological(jid):
    with jobs_lock: j=jobs.get(jid)
    if not j or j["status"]!="done": return jsonify({"error":"Not ready"}),404
    return jsonify(j["result"].get("physiological",{}))

@app.route("/analysis/report/<jid>")
def analysis_report(jid):
    with jobs_lock: j=jobs.get(jid)
    if not j or j["status"]!="done": return jsonify({"error":"Not ready"}),404
    return jsonify(j["result"].get("analytical_report",{}))

@app.route("/analysis/keyframes/<jid>")
def analysis_keyframes(jid):
    with jobs_lock: j=jobs.get(jid)
    if not j or j["status"]!="done": return jsonify({"error":"Not ready"}),404
    return jsonify(j["result"].get("keyframes",[]))

# ─── Chat ─────────────────────────────────────────────────

@app.route("/chat/<video_hash>", methods=["POST"])
def chat(video_hash):
    data=request.get_json(force=True,silent=True) or {}
    query=(data.get("query") or "").strip()
    if not query: return jsonify({"error":"query required"}),400
    top_k=int(data.get("top_k",3)); use_vlm=bool(data.get("use_vlm",False))
    if video_hash not in _chat_indexes:
        idx=VideoChatIndex(video_hash)
        if not idx.load(): return jsonify({"error":"Index not found"}),404
        _chat_indexes[video_hash]=idx
    return jsonify(_chat_indexes[video_hash].chat(query,top_k=top_k,use_vlm=use_vlm))

@app.route("/chat_by_job/<jid>", methods=["POST"])
def chat_by_job(jid):
    with jobs_lock: j=jobs.get(jid)
    if not j: return jsonify({"error":"Job not found"}),404
    if j["status"]!="done": return jsonify({"error":"Not ready","status":j["status"]}),202
    vh=j.get("video_hash") or j["result"].get("video_hash","")
    if not vh: return jsonify({"error":"No video hash"}),500
    return chat(vh)

# ─── Translation + download ───────────────────────────────

@app.route("/translate/<lang>")
def translate(lang):
    if lang not in SUPPORTED_LANGUAGES: return jsonify({"error":f"Unsupported: {lang}"}),400
    t=session.get("transcript","")
    if not t: return jsonify({"error":"No transcript"}),400
    tr=session.get("translations",{})
    if lang in tr: return jsonify({"text":tr[lang]})
    try:
        translated=GoogleTranslator(source="auto",target=SUPPORTED_LANGUAGES[lang]).translate(t)
        tr[lang]=translated; session["translations"]=tr; return jsonify({"text":translated})
    except Exception as e: return jsonify({"error":str(e)}),500

@app.route("/translate_to_english/<lang>")
def translate_to_english(lang):
    tr=session.get("translations",{}); src=tr.get(lang,"")
    if not src: return jsonify({"error":"No translation"}),400
    key=f"{lang}_back_en"
    if key in tr: return jsonify({"text":tr[key]})
    try:
        back=GoogleTranslator(source="auto",target="en").translate(src)
        tr[key]=back; session["translations"]=tr; return jsonify({"text":back})
    except Exception as e: return jsonify({"error":str(e)}),500

@app.route("/translate_to_braille/<lang>")
def translate_to_braille(lang):
    tr=session.get("translations",{}); src=tr.get(lang,"")
    if not src: return jsonify({"error":"No translation"}),400
    return jsonify({"text":_to_braille(src)})

@app.route("/download/<dtype>")
def download(dtype):
    if dtype=="original":   text=session.get("transcript",""); fn="transcript.txt"
    elif dtype=="braille":  text=session.get("braille",""); fn="transcript_braille.txt"
    elif dtype in SUPPORTED_LANGUAGES:
        text=session.get("translations",{}).get(dtype,""); fn=f"transcript_{dtype.lower()}.txt"
        if not text: return "Translation not generated.",400
    else: return "Unknown type.",400
    return Response(text,mimetype="text/plain; charset=utf-8",
                    headers={"Content-Disposition":f'attachment; filename="{fn}"'})

@app.route("/uploads/<path:fn>")
def uploads(fn): return send_from_directory(app.config["UPLOAD_FOLDER"],fn,as_attachment=False)

if __name__=="__main__":
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  VideoMind — Mode-Aware Pipeline                         ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  Mode 1: video_to_text  — Whisper + CLIP + chatbot       ║")
    print("║  Mode 2: forensic       — Full pipeline, all GPUs        ║")
    print("║  Mode 3: live           — CLIP+deepfake+rPPG+emotion     ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  Webcam: /webcam/start  /webcam/frame  /webcam/stop      ║")
    print("║  Results: /results/{text,forensic,live}/<jid>            ║")
    print("╚══════════════════════════════════════════════════════════╝")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)