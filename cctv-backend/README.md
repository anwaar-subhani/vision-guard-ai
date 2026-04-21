# CCTV Backend (FastAPI) — CCTV Anomaly Detection + MongoDB

This backend accepts video uploads, runs selected anomaly detectors in parallel,
streams detections via Server-Sent Events, and stores video metadata + detections in MongoDB.

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

4. Create `cctv-backend/.env`:

```env
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=cctv
AUTO_DELETE_UPLOADS=false
```

5. Run the server:

```powershell
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

6. Test processing (SSE):

```powershell
curl -N -F "file=@C:\path\to\video.mp4" -F "anomaly_types=[\"gunshot_audio\"]" http://127.0.0.1:8000/process-video
```

Useful APIs:
- `GET /health` → includes MongoDB status
- `POST /process-video` → upload + stream detection events
- `POST /process-video-realtime` → unified playback-synced realtime endpoint (mode-driven)
- `POST /process-video-realtime-gunshot` → playback-synced gunshot processing (SSE tick + event stream)
- `POST /process-video-realtime-fire` → playback-synced fire/explosion processing (SSE tick + prediction + event)
- `POST /process-video-realtime-fall` → playback-synced sudden-fall processing (SSE tick + prediction + event)
- `POST /process-video-realtime-fight` → playback-synced fight processing (SSE tick + prediction + event)
- `GET /stats/overview` → dashboard totals + anomaly breakdown + recent videos
- `GET /stats/trends?days=7` → daily trends for videos/detections
- `GET /alerts?status=all|active|investigating|resolved&limit=50` → alert feed for Alerts page
- `GET /analytics/summary?days=7` → trends + time patterns + hotspot summary for Analytics page

Realtime gunshot SSE message flow (`POST /process-video-realtime-gunshot`):
- `video_meta` → IDs + stream URLs
- `tick` → every audio window (time, end_time, confidence)
- `event` → emitted only when confidence crosses gunshot threshold
- `error` → detector/runtime error
- `done` → final message

Realtime fire SSE message flow (`POST /process-video-realtime-fire`):
- `video_meta` → IDs + stream URLs
- `tick` → every sampled frame window (time, end_time, confidence)
- `prediction` → continuous label stream (`Explosion/Fire` or `No Fire`)
- `event` → emitted when fire confidence crosses threshold
- `error` → detector/runtime error
- `done` → final message

Unified realtime endpoint (`POST /process-video-realtime`):
- Form fields:
	- `file`: video file
	- `mode`: one of `gunshot`, `fire`, `fall`, `fight`
- The endpoint uses one internal pipeline and mode config table (`REALTIME_MODE_CONFIG`).
- To add future realtime models (e.g., crowd/scream), add one entry in that mode config
	with anomaly id, stream function, default labels, threshold, and stream kwargs.

Realtime fall/fight SSE message flow (`POST /process-video-realtime-fall` and `/process-video-realtime-fight`):
- `video_meta` → IDs + stream URLs
- `tick` → timeline progress for each model window
- `prediction` → continuous label stream (`Sudden Fall`/`Normal Posture`, `Fight`/`No Fight`)
- `event` → emitted when model confidence crosses threshold
- `error` → detector/runtime error
- `done` → final message

Explosion/Fire model integration:
- Frontend toggle id `explosion_fire_visual` now runs `detectors/fire_detector.py`.
- Fire detector expects model weights at `models/fire_model_best.pt`.
- Required dependency: `ultralytics` (included in `requirements.txt`).
- Events are streamed during processing via `POST /process-video` in near-real-time.

Notes:
- Uploaded files are stored in `uploads/`.
- Set `AUTO_DELETE_UPLOADS=true` if you want files removed after processing.
- If `MONGODB_URI` is not set, processing still works but stats endpoints return 503.
