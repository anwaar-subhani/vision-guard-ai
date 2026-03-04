"""
Simple gunshot detection helper.

Behavior:
- Attempts to use a user-supplied model file if present at `model_path` (placeholder hook).
- Otherwise falls back to extracting audio via `ffmpeg` and a short-time RMS peak detector
  implemented with `wave` and `numpy`.

Notes:
- This is a pragmatic local/testing implementation. For production you should replace the
  fallback with your trained model integration (PyTorch/TensorFlow) in `model_predict_wrapper`.
"""
import os
import subprocess
import wave
import numpy as np
from typing import List, Dict


def is_ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


def extract_audio(video_path: str, out_wav: str = None, sample_rate: int = 16000) -> str:
    """Extract mono 16-bit WAV from a video using ffmpeg. Returns path to WAV file."""
    if out_wav is None:
        base = os.path.splitext(video_path)[0]
        out_wav = f"{base}_audio.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-sample_fmt",
        "s16",
        "-f",
        "wav",
        out_wav,
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out_wav


def read_wav(wav_path: str):
    with wave.open(wav_path, "rb") as wf:
        nframes = wf.getnframes()
        framerate = wf.getframerate()
        frames = wf.readframes(nframes)

    # ffmpeg is asked to produce s16 -> 16-bit signed integers
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, framerate


def simple_gunshot_detection(audio: np.ndarray, sr: int) -> List[Dict]:
    """Very lightweight energy-based detector.

    Returns a list of events: {time: seconds, confidence: 0-100}
    """
    if audio.size == 0:
        return []

    win_size = int(0.2 * sr)  # 200 ms windows
    hop = int(0.1 * sr)  # 100 ms hop
    energies = []
    times = []
    for start in range(0, max(1, len(audio) - win_size + 1), hop):
        w = audio[start : start + win_size]
        rms = np.sqrt(np.mean(w * w) + 1e-12)
        energies.append(rms)
        times.append(start / sr)

    energies = np.array(energies)
    if energies.size == 0:
        return []

    median = float(np.median(energies))
    max_e = float(energies.max())

    # Heuristic threshold: either a multiple of median or an absolute floor
    threshold = max(median * 8.0, 0.25)

    events: List[Dict] = []
    i = 0
    while i < len(energies):
        if energies[i] > threshold:
            j = i
            peak = energies[i]
            peak_idx = i
            while j + 1 < len(energies) and energies[j + 1] > (threshold * 0.5):
                j += 1
                if energies[j] > peak:
                    peak = energies[j]
                    peak_idx = j

            timestamp = times[peak_idx]
            # confidence normalized to 0..1
            denom = max_e - median + 1e-9
            confidence = float(min(1.0, (peak - median) / denom)) if denom > 0 else 1.0
            events.append({"time": round(timestamp, 2), "confidence": round(confidence * 100.0, 1)})
            i = j + 1
        else:
            i += 1

    return events


def model_predict_wrapper(model, audio: np.ndarray, sr: int) -> List[Dict]:
    """Placeholder wrapper for user-supplied model.

    If you have a trained model, implement how to transform `audio` into model inputs and
    return a list of events similar to the fallback detector.
    """
    # If model provides a `predict` method, call it
    if hasattr(model, "predict"):
        return model.predict(audio, sr)

    # Try a basic PyTorch-based inference pattern (best-effort placeholder)
    try:
        import torch

        model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(audio).float().unsqueeze(0)
            out = model(tensor)
            # Model-specific postprocessing goes here. Return empty list by default.
            return []
    except Exception:
        return []


def detect_gunshots(video_path: str, model_path: str = None) -> List[Dict]:
    """Detect gunshot events in a video file.

    If `model_path` exists, the code attempts to use it (placeholder). Otherwise the
    fallback extractor + energy detector is used.
    """
    # If a model exists, try to use it (user must supply model implementation)
    if model_path and os.path.exists(model_path):
        try:
            # Example: try to load PyTorch model
            import torch

            model = torch.load(model_path, map_location="cpu")
            wav = extract_audio(video_path)
            audio, sr = read_wav(wav)
            return model_predict_wrapper(model, audio, sr)
        except Exception:
            # If any error, continue to fallback
            pass

    # Fallback implementation
    if not is_ffmpeg_available():
        raise RuntimeError("ffmpeg not found on PATH. Install ffmpeg to enable audio extraction.")

    wav = extract_audio(video_path)
    audio, sr = read_wav(wav)
    events = simple_gunshot_detection(audio, sr)
    return events
