"""
Crowd gathering detection (visual-based).

Stub — returns empty results until a trained model is integrated.
Place your model file at models/crowd_model.pth and implement inference below.
"""

def detect(video_path: str, model_dir: str) -> list[dict]:
    """Detect unusual crowd gathering events in a video.

    Returns a list of event dicts:
        {"time": <seconds>, "confidence": <0-100>, "label": "Crowd Gathering"}
    """
    # TODO: Load your crowd model from model_dir and run inference on video_path.
    return []
