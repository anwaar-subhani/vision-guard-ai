import sys
from pathlib import Path

# Ensure this file's directory is on sys.path so `detectors` package is found
# regardless of the working directory uvicorn is launched from.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import shutil
import uuid
import json
import os
import queue
import threading

from detectors import DETECTOR_REGISTRY

app = FastAPI(title="CCTV Backend – Anomaly Detection")

# Allow the Vite dev server (and any localhost origin) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_DIR = BASE_DIR / "models"

UPLOAD_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Process video – runs all selected anomaly detectors concurrently
# ---------------------------------------------------------------------------

@app.post("/process-video")
async def process_video(
    file: UploadFile = File(...),
    anomaly_types: str = Form(...),  # JSON-encoded list of anomaly IDs
):
    """
    Upload a video and stream detection results as Server-Sent Events.

    Each SSE message is a JSON object with a "type" field:
      - {"type":"event", "anomalyId":"…", "time":…, "confidence":…, "label":"…"}
      - {"type":"error", "anomalyId":"…", "message":"…"}
      - {"type":"detector_done", "anomalyId":"…"}
      - {"type":"done"}  (final message)
    """

    # ---- parse & validate anomaly selection --------------------------------
    try:
        selected: list[str] = json.loads(anomaly_types)
    except (json.JSONDecodeError, TypeError):
        raise HTTPException(status_code=400, detail="anomaly_types must be a JSON array of strings.")

    if not selected:
        raise HTTPException(status_code=400, detail="Select at least one anomaly type.")

    unknown = [a for a in selected if a not in DETECTOR_REGISTRY]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown anomaly types: {unknown}")

    # ---- save uploaded file ------------------------------------------------
    filename = f"{uuid.uuid4().hex}_{Path(file.filename).name}"
    dest_path = UPLOAD_DIR / filename

    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        try:
            file.file.close()
        except Exception:
            pass

    # ---- SSE streaming generator -------------------------------------------
    def sse_generator():
        q: queue.Queue = queue.Queue()

        def run_detector(anomaly_id: str):
            detect_fn = DETECTOR_REGISTRY[anomaly_id]
            try:
                # Works with generators (yield) and plain lists (return [])
                for event in detect_fn(str(dest_path), str(MODEL_DIR)):
                    q.put(json.dumps({
                        "type": "event",
                        "anomalyId": anomaly_id,
                        **event,
                    }))
            except Exception as e:
                q.put(json.dumps({
                    "type": "error",
                    "anomalyId": anomaly_id,
                    "message": str(e),
                }))
            q.put(json.dumps({"type": "detector_done", "anomalyId": anomaly_id}))

        # Launch all detectors in parallel threads
        threads = []
        for aid in selected:
            t = threading.Thread(target=run_detector, args=(aid,), daemon=True)
            t.start()
            threads.append(t)

        finished = 0
        total = len(selected)
        while finished < total:
            try:
                data = q.get(timeout=300)
                parsed = json.loads(data)
                if parsed.get("type") == "detector_done":
                    finished += 1
                yield f"data: {data}\n\n"
            except queue.Empty:
                break

        yield 'data: {"type":"done"}\n\n'

        # Wait for threads then clean up the uploaded file
        for t in threads:
            t.join(timeout=5)
        try:
            os.remove(dest_path)
        except OSError:
            pass

    return StreamingResponse(sse_generator(), media_type="text/event-stream")
