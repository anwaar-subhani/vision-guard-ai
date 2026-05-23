"""FastAPI app entry point for the CCTV backend."""

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure this file's directory is on sys.path so local packages are found
# regardless of where uvicorn is launched from.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from api.routes.health_routes import router as health_router
from api.routes.stats_routes import router as stats_router
from api.routes.videos_routes import router as videos_router
from api.routes.alerts_routes import router as alerts_router
from api.routes.analytics_routes import router as analytics_router
from api.routes.process_routes import router as process_router
from core import app_state as st
from detectors.scream_detector import preload_models as preload_scream_models
from detectors.fire_detector import preload_model as preload_fire_model
from detectors.crowd_detector import preload_model as preload_crowd_model

app = FastAPI(title="CCTV Backend – Anomaly Detection")

ALL_ROUTERS = [
    health_router,
    stats_router,
    videos_router,
    alerts_router,
    analytics_router,
    process_router,
]

# Allow frontend (and localhost origins) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for router in ALL_ROUTERS:
    app.include_router(router)


@app.on_event("startup")
async def _warmup_models() -> None:
    """Best-effort model warmup to reduce first realtime inference latency."""
    ok, message = preload_scream_models(str(st.MODEL_DIR))
    print(f"[startup] {message}")
    ok, message = preload_fire_model(str(st.MODEL_DIR))
    print(f"[startup] {message}")
    ok, message = preload_crowd_model(str(st.MODEL_DIR))
    print(f"[startup] {message}")
