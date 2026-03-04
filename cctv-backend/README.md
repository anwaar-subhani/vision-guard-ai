# CCTV Backend (FastAPI) — Gunshot Detection

This folder contains a minimal FastAPI backend to accept video uploads and run a simple
gunshot detection flow. It includes a fallback detector (ffmpeg + energy-based) and a
hook where you can integrate your trained model.

Quick start (Windows):

1. Install Python 3.9+
2. Install ffmpeg and ensure `ffmpeg` is on your PATH
3. Create a virtual environment and install dependencies:

```powershell
cd cctv-backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

4. Run the server:

```powershell
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

5. Test upload (use path to a local MP4):

```powershell
curl -F "file=@C:\path\to\video.mp4" http://127.0.0.1:8000/upload
```

Notes:
- Uploads are stored in the `uploads/` directory created next to `main.py`.
- If you have a trained model, place it at `models/gunshot_model.pt` and implement
  `model_predict_wrapper` in `gunshot_detector.py` to return events.
