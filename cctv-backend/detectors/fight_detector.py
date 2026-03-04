"""Fight detection (visual-based) using sliding-window video inference.

This detector is intentionally aligned with your provided `test.py` flow:
- loads `best_fight_model.pth`
- uses helpers from `fight_model.py` (`NUM_FRAMES`, `FRAME_SIZE`, `DEVICE`,
  `get_transforms`, `get_improved_model`)
- runs sliding-window inference and yields events in real time for SSE.
"""

from __future__ import annotations

import os
import math
import importlib
from typing import Dict, List, Tuple

import cv2
import torch
import numpy as np
from PIL import Image


# ===========================================================================
# CONFIGURATION — tweak these values as needed
# ===========================================================================

MODEL_FILENAME = "best_fight_model.pth"
CONF_THRESHOLD = 0.50
STRIDE_DIVISOR = 3          # stride = NUM_FRAMES // STRIDE_DIVISOR
WINDOW_BATCH_SIZE = 4       # number of clips per model forward pass
MIN_EVENT_GAP_SEC = 1.0     # suppress near-duplicate events from overlap


# ---------------------------------------------------------------------------
# Cached model bundle
# ---------------------------------------------------------------------------

_cached_model = None
_cached_transforms = None
_cached_model_path: str | None = None
_cached_device = None
_cached_num_frames: int | None = None
_cached_frame_size: int | None = None


def _sliding_window_indices(total_frames: int, window_size: int, stride: int) -> List[Tuple[int, int]]:
    """Generate start/end indices for sliding windows."""
    indices: List[Tuple[int, int]] = []
    if total_frames <= 0:
        return indices

    if total_frames < window_size:
        return [(0, total_frames)]

    for start in range(0, total_frames - window_size + 1, stride):
        indices.append((start, start + window_size))

    if indices and indices[-1][1] < total_frames:
        indices.append((total_frames - window_size, total_frames))

    return indices


def _preprocess_clip(frames_rgb: List[np.ndarray], transform, device, num_frames: int) -> torch.Tensor:
    """Convert list of RGB numpy frames to model input tensor (1, C, T, H, W)."""
    clip = list(frames_rgb)
    if not clip:
        raise ValueError("Empty clip encountered during fight preprocessing.")

    # pad edge windows by repeating last frame
    while len(clip) < num_frames:
        clip.append(clip[-1])

    processed = []
    for frame in clip[:num_frames]:
        pil_img = Image.fromarray(frame)
        processed.append(transform(pil_img))

    tensor = torch.stack(processed, dim=1).unsqueeze(0)  # (1, C, T, H, W)
    return tensor.to(device, dtype=torch.float32)


def _load_fight_bundle(model_path: str):
    """Load model + transforms + metadata from user's training helper module."""
    global _cached_model
    global _cached_transforms
    global _cached_model_path
    global _cached_device
    global _cached_num_frames
    global _cached_frame_size

    if _cached_model is not None and _cached_model_path == model_path:
        return (
            _cached_model,
            _cached_transforms,
            _cached_device,
            _cached_num_frames,
            _cached_frame_size,
        )

    try:
        model_module = importlib.import_module("fight_model")
    except Exception as exc:
        raise RuntimeError(
            "Fight detector requires `cctv-backend/fight_model.py` with: "
            "NUM_FRAMES, FRAME_SIZE, DEVICE, get_transforms, get_improved_model. "
            f"Import error: {exc}"
        )

    required = ["NUM_FRAMES", "FRAME_SIZE", "DEVICE", "get_transforms", "get_improved_model"]
    missing = [name for name in required if not hasattr(model_module, name)]
    if missing:
        raise RuntimeError(
            "fight_model.py is missing required symbols for fight detection: "
            f"{missing}."
        )

    num_frames = int(model_module.NUM_FRAMES)
    frame_size = int(model_module.FRAME_SIZE)
    device = model_module.DEVICE
    transform = model_module.get_transforms("val")
    model = model_module.get_improved_model(num_classes=2)

    state = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    try:
        model.load_state_dict(state)
    except RuntimeError:
        # handle DataParallel checkpoints with "module." prefix
        if isinstance(state, dict):
            fixed = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state.items()}
            model.load_state_dict(fixed)
        else:
            raise

    model = model.to(device)
    model.eval()

    _cached_model = model
    _cached_transforms = transform
    _cached_model_path = model_path
    _cached_device = device
    _cached_num_frames = num_frames
    _cached_frame_size = frame_size

    return model, transform, device, num_frames, frame_size


def detect(video_path: str, model_dir: str):
    """Yield fight events in real time for SSE streaming.

    Yields event dicts:
        {"time": <seconds>, "confidence": <0-100>, "label": "Fight"}
    """
    model_path = os.path.join(model_dir, MODEL_FILENAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Fight model not found at {model_path}. "
            f"Place your .pth file as {MODEL_FILENAME} inside models/."
        )

    model, transform, device, num_frames, frame_size = _load_fight_bundle(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames_rgb: List[np.ndarray] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (frame_size, frame_size))
            frames_rgb.append(resized)

        if not frames_rgb:
            return

        stride = max(1, int(num_frames // STRIDE_DIVISOR))
        windows = _sliding_window_indices(len(frames_rgb), num_frames, stride)
        gap_frames = max(1, int(math.ceil(MIN_EVENT_GAP_SEC * fps)))
        last_emit_frame = -10_000_000

        with torch.no_grad():
            for batch_start in range(0, len(windows), WINDOW_BATCH_SIZE):
                batch_windows = windows[batch_start: batch_start + WINDOW_BATCH_SIZE]

                clips = []
                for start, end in batch_windows:
                    clip_frames = frames_rgb[start:end]
                    clips.append(_preprocess_clip(clip_frames, transform, device, num_frames))

                # (B, C, T, H, W)
                inputs = torch.cat(clips, dim=0)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()

                for i, (start, end) in enumerate(batch_windows):
                    conf = float(probs[i])
                    if conf < CONF_THRESHOLD:
                        continue

                    # debounce overlapping positive windows
                    if (start - last_emit_frame) < gap_frames:
                        continue

                    last_emit_frame = start
                    yield {
                        "time": round(start / fps, 2),
                        "confidence": round(conf * 100.0, 1),
                        "label": "Fight",
                    }
    finally:
        cap.release()
