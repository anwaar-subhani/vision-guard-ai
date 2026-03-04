"""
Gunshot detection (audio-based) using a trained ResNet-101 classifier.

Pipeline:
1. Extract audio from video via ffmpeg
2. Load audio with librosa (16 kHz mono)
3. Slice into overlapping 2-second frames (1.5 s step)
4. Convert each frame -> mel-spectrogram -> 3-channel 224x224 image
5. Run through ResNet-101 classifier (class 0 = no gunshot, class 1 = gunshot)
6. Return detected events with timestamps and confidence

Model file expected at: models/gunshot_trained_model.pth
"""

import os
import subprocess
import numpy as np
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torchvision.models as models
import librosa
import imageio_ffmpeg


# ===========================================================================
# CONFIGURATION — change these values to tune the detector
# ===========================================================================

MODEL_FILENAME       = "gunshot_trained_model.pth"   # model weights file inside models/
SAMPLE_RATE          = 16000       # audio sample rate (Hz)
FRAME_DURATION       = 2.0         # length of each audio frame (seconds)
STRIDE               = 0.25       # step between frames (seconds) — lower = finer but slower
BATCH_SIZE           = 32          # number of frames per inference batch
CONFIDENCE_THRESHOLD = 0.7         # minimum probability to count as a gunshot
N_MELS               = 128         # mel-spectrogram frequency bins
HOP_LENGTH           = 512         # STFT hop length
N_FFT                = 2048        # STFT window size
IMAGE_SIZE           = 224         # model input resolution (pixels)


# ---------------------------------------------------------------------------
# Model architecture (must match training code exactly)
# ---------------------------------------------------------------------------

class ResNet101Classifier(nn.Module):
    """ResNet-101 based classifier for gunshot detection."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()

        try:
            if pretrained:
                resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
            else:
                resnet = models.resnet101(weights=None)
        except (AttributeError, TypeError):
            resnet = models.resnet101(pretrained=pretrained)

        self.features = nn.Sequential(*list(resnet.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _auto_device() -> torch.device:
    """Pick the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except AttributeError:
        pass
    return torch.device("cpu")


def _load_model(model_path: str, device: torch.device) -> nn.Module:
    """Load a ResNet101Classifier checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model = ResNet101Classifier(num_classes=2, pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model = ResNet101Classifier(num_classes=2, pretrained=False)
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        model = ResNet101Classifier(num_classes=2, pretrained=False)
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def _extract_audio_ffmpeg(video_path: str, sample_rate: int = 16000) -> str:
    """Extract mono 16-bit WAV from video using the bundled ffmpeg binary."""
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    out_wav = os.path.splitext(video_path)[0] + "_audio.wav"
    cmd = [
        ffmpeg_path, "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-sample_fmt", "s16", "-f", "wav", out_wav,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out_wav


def _preprocess_frame(audio_frame: np.ndarray, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Convert a 1-D audio frame into a (3, IMAGE_SIZE, IMAGE_SIZE) mel-spectrogram tensor."""
    mel = librosa.feature.melspectrogram(
        y=audio_frame, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    vmin, vmax = mel_db.min(), mel_db.max()
    mel_db = (mel_db - vmin) / (vmax - vmin + 1e-8)

    mel_3ch = np.stack([mel_db] * 3, axis=0)  # (3, N_MELS, T)
    tensor = torch.from_numpy(mel_3ch).float()

    tensor = nn.functional.interpolate(
        tensor.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False,
    ).squeeze(0)

    return tensor


def _extract_frames(
    audio: np.ndarray, sr: int, frame_dur: float = FRAME_DURATION, step_dur: float = STRIDE,
) -> List[Tuple[float, float, np.ndarray]]:
    """Slice audio into overlapping frames."""
    frame_samples = int(frame_dur * sr)
    step_samples = int(step_dur * sr)
    total = len(audio)
    frames: List[Tuple[float, float, np.ndarray]] = []
    start = 0
    while start < total:
        end = min(start + frame_samples, total)
        chunk = audio[start:end]
        if len(chunk) < frame_samples:
            chunk = np.pad(chunk, (0, frame_samples - len(chunk)), mode="constant")
        frames.append((start / sr, end / sr, chunk))
        start += step_samples
        if end >= total:
            break
    return frames


# ---------------------------------------------------------------------------
# Singleton cache so the model is loaded once across requests
# ---------------------------------------------------------------------------
_cached_model: nn.Module | None = None
_cached_model_path: str | None = None
_device: torch.device | None = None


def _get_model(model_path: str) -> Tuple[nn.Module, torch.device]:
    global _cached_model, _cached_model_path, _device
    if _cached_model is not None and _cached_model_path == model_path:
        return _cached_model, _device  # type: ignore[return-value]
    _device = _auto_device()
    _cached_model = _load_model(model_path, _device)
    _cached_model_path = model_path
    return _cached_model, _device


# ---------------------------------------------------------------------------
# Public entry point (matches detector registry signature)
# ---------------------------------------------------------------------------


def detect(video_path: str, model_dir: str):
    """Detect gunshot events in a video file.

    This is a **generator** – it yields event dicts as soon as each batch is
    processed so the SSE endpoint can stream them to the frontend in real time.

    Yields:
        dict: {"time": <s>, "confidence": <0-100>, "label": "Gunshot"}
    """
    model_path = os.path.join(model_dir, MODEL_FILENAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Gunshot model not found at {model_path}. "
            f"Place your .pth file as {MODEL_FILENAME} inside the models/ directory."
        )

    # 1. Extract audio from video
    wav_path = _extract_audio_ffmpeg(video_path, SAMPLE_RATE)

    try:
        # 2. Load audio
        audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)

        # 3. Split into frames
        frames = _extract_frames(audio, sr)

        # 4. Load model (cached across requests)
        model, device = _get_model(model_path)

        # 5. Batched inference – yield events per batch for real-time streaming
        num_frames = len(frames)

        for batch_start in range(0, num_frames, BATCH_SIZE):
            batch_frames = frames[batch_start : batch_start + BATCH_SIZE]
            tensors = [_preprocess_frame(chunk, sr) for (_, _, chunk) in batch_frames]
            batch_tensor = torch.stack(tensors, dim=0).to(device)  # (B, 3, 224, 224)

            with torch.no_grad():
                logits = model(batch_tensor)
                probs = torch.softmax(logits, dim=1)
                gunshot_probs = probs[:, 1].cpu().numpy()

            for i, (start_t, end_t, _) in enumerate(batch_frames):
                prob = float(gunshot_probs[i])
                if prob > CONFIDENCE_THRESHOLD:
                    yield {
                        "time": round(start_t, 2),
                        "confidence": round(prob * 100.0, 1),
                        "label": "Gunshot",
                    }
    finally:
        # Clean up temporary wav even if generator is abandoned
        try:
            os.remove(wav_path)
        except OSError:
            pass
