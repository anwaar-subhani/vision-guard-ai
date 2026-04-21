from pathlib import Path
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pymongo.errors import PyMongoError

from core import app_state as st

router = APIRouter(tags=["videos"])


def _parse_object_id(raw_id: str, error_message: str):
    try:
        return ObjectId(raw_id)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=400, detail=error_message)


def _serialize_detection(doc: dict[str, Any]) -> dict[str, Any]:
    event_time = float(doc.get("time", 0) or 0)
    created_at = st.ensure_datetime(doc.get("created_at"))
    return {
        "id": str(doc.get("_id")),
        "anomaly_id": str(doc.get("anomaly_id") or "unknown"),
        "label": str(doc.get("label") or doc.get("anomaly_id") or "Anomaly"),
        "time": event_time,
        "end_time": float(doc.get("end_time")) if doc.get("end_time") is not None else None,
        "confidence": float(doc.get("confidence", 0) or 0),
        "bbox": doc.get("bbox"),
        "created_at": created_at.isoformat() if created_at else None,
        "video_time": st.format_video_time(event_time),
    }


def _serialize_video(video_id: str, video: dict[str, Any]) -> dict[str, Any]:
    created_at = st.ensure_datetime(video.get("created_at"))
    completed_at = st.ensure_datetime(video.get("completed_at"))
    processed_path = str(video.get("processed_path") or "").strip()

    return {
        "id": str(video.get("_id")),
        "filename": str(video.get("original_filename") or video.get("stored_filename") or "Unknown video"),
        "status": str(video.get("status") or "unknown"),
        "created_at": created_at.isoformat() if created_at else None,
        "completed_at": completed_at.isoformat() if completed_at else None,
        "total_detections": int(video.get("total_detections", 0) or 0),
        "selected_anomalies": list(video.get("selected_anomalies") or []),
        "processed_stream_url": f"/videos/{video_id}/processed-stream",
        "has_processed_video": bool(processed_path) and Path(processed_path).exists(),
    }


@router.get("/videos/{video_id}/detections")
def video_detections(video_id: str) -> dict[str, Any]:
    videos, detections = st.require_db()
    oid = _parse_object_id(video_id, "Invalid video id")

    video = videos.find_one(
        {"_id": oid},
        {
            "original_filename": 1,
            "stored_filename": 1,
            "status": 1,
            "created_at": 1,
            "completed_at": 1,
            "total_detections": 1,
            "selected_anomalies": 1,
        },
    )

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    docs = list(
        detections.find(
            {"video_id": oid},
            {
                "anomaly_id": 1,
                "label": 1,
                "time": 1,
                "end_time": 1,
                "confidence": 1,
                "bbox": 1,
                "created_at": 1,
            },
        ).sort("time", 1)
    )

    items = [_serialize_detection(doc) for doc in docs]

    return {
        "video": _serialize_video(video_id, video),
        "detections": items,
    }


@router.get("/videos/{video_id}/stream")
def stream_video(video_id: str):
    videos, _ = st.require_db()
    oid = _parse_object_id(video_id, "Invalid video id")

    video = videos.find_one({"_id": oid}, {"upload_path": 1, "stored_filename": 1})
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    source_path = st.resolve_source_video_path(video)
    return FileResponse(path=str(source_path), media_type="video/mp4")


@router.get("/videos/{video_id}/processed-stream")
def stream_processed_video(video_id: str, force_regenerate: bool = False):
    videos, detections = st.require_db()
    oid = _parse_object_id(video_id, "Invalid video id")

    video = videos.find_one(
        {"_id": oid},
        {
            "upload_path": 1,
            "stored_filename": 1,
            "processed_path": 1,
        },
    )
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    existing_processed = Path(str(video.get("processed_path") or "").strip()) if video.get("processed_path") else None
    if existing_processed and existing_processed.exists() and not force_regenerate:
        return FileResponse(path=str(existing_processed), media_type="video/mp4")

    source_path = st.resolve_source_video_path(video)
    out_path = st.PROCESSED_DIR / f"processed_{video_id}.mp4"

    event_docs = list(
        detections.find(
            {"video_id": oid},
            {
                "anomaly_id": 1,
                "label": 1,
                "time": 1,
                "confidence": 1,
                "bbox": 1,
            },
        ).sort("time", 1)
    )

    events: list[dict[str, Any]] = []
    for row in event_docs:
        events.append(
            {
                "anomaly_id": str(row.get("anomaly_id") or "unknown"),
                "label": str(row.get("label") or row.get("anomaly_id") or "Anomaly"),
                "time": st.safe_float(row.get("time"), 0.0),
                "confidence": st.safe_float(row.get("confidence"), 0.0),
                "bbox": row.get("bbox"),
            }
        )

    try:
        st.generate_processed_video_file(source_path=source_path, output_path=out_path, events=events)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to render processed video: {exc}")

    try:
        videos.update_one(
            {"_id": oid},
            {
                "$set": {
                    "processed_path": str(out_path),
                    "processed_updated_at": st.now_utc(),
                }
            },
        )
    except PyMongoError:
        pass

    return FileResponse(path=str(out_path), media_type="video/mp4")
