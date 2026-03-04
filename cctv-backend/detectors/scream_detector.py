"""
Scream detection (audio-based).

Stub — returns empty results until a trained model is integrated.
Place your model file at models/scream_model.pth and implement inference below.
"""

from typing import List, Dict


def detect(video_path: str, model_dir: str) -> List[Dict]:
    """Detect scream / distress call events in a video's audio track.

    Returns a list of event dicts:
        {"time": <seconds>, "confidence": <0-100>, "label": "Scream"}
    """
    # TODO: Load model from model_dir and run inference on extracted audio
    return []
