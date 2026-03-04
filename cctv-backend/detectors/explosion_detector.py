"""
Explosion / Fire detection (visual-based).

Stub — returns empty results until a trained model is integrated.
Place your model file at models/explosion_model.pth and implement inference below.
"""

from typing import List, Dict


def detect(video_path: str, model_dir: str) -> List[Dict]:
    """Detect explosion or fire events in a video.

    Returns a list of event dicts:
        {"time": <seconds>, "confidence": <0-100>, "label": "Explosion/Fire"}
    """
    # TODO: Load model from model_dir and run inference on video_path
    return []
