import re
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, HTTPException

from core import app_state as st

router = APIRouter(tags=["alerts"])


def _parse_object_id(raw_id: str, error_message: str):
    try:
        return ObjectId(raw_id)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=400, detail=error_message)


def _alert_status(doc: dict[str, Any]) -> str:
    resolved_at = st.ensure_datetime(doc.get("resolved_at"))
    resolution_status = str(doc.get("resolution_status") or "").lower()
    return "resolved" if resolution_status == "resolved" or resolved_at else "active"


def _serialize_alert(doc: dict[str, Any]) -> dict[str, Any]:
    confidence = float(doc.get("confidence", 0) or 0)
    event_time = float(doc.get("time", 0) or 0)
    created_at = st.ensure_datetime(doc.get("created_at"))
    resolved_at = st.ensure_datetime(doc.get("resolved_at"))
    label = str(doc.get("label") or doc.get("anomaly_id") or "Anomaly")

    return {
        "id": str(doc.get("_id")),
        "video_id": str(doc.get("video_id")) if doc.get("video_id") else None,
        "type": label,
        "anomaly_id": str(doc.get("anomaly_id") or "unknown"),
        "description": f"{label} detected in analyzed video",
        "filename": str(doc.get("filename") or "Unknown video"),
        "timestamp": created_at.isoformat() if created_at else None,
        "resolved_at": resolved_at.isoformat() if resolved_at else None,
        "severity": st.confidence_to_severity(confidence),
        "status": _alert_status(doc),
        "confidence": round(confidence, 1),
        "video_time_seconds": event_time,
        "video_time": st.format_video_time(event_time),
    }


@router.get("/alerts")
def get_alerts(
    limit: int = 50,
    status: str | None = None,
    video_id: str | None = None,
    video_search: str | None = None,
) -> dict[str, Any]:
    _, detections = st.require_db()

    limit = max(1, min(limit, 500))

    match_stage: dict[str, Any] = {}
    if video_id:
        try:
            match_stage["video_id"] = ObjectId(video_id)
        except (InvalidId, TypeError):
            raise HTTPException(status_code=400, detail="Invalid video id")

    pipeline = [
        {"$match": match_stage} if match_stage else {"$match": {}},
        {"$sort": {"created_at": -1}},
        {"$limit": limit},
        {
            "$lookup": {
                "from": "videos",
                "localField": "video_id",
                "foreignField": "_id",
                "as": "video",
            }
        },
        {"$unwind": {"path": "$video", "preserveNullAndEmptyArrays": True}},
        {
            "$project": {
                "video_id": 1,
                "anomaly_id": 1,
                "label": 1,
                "time": 1,
                "confidence": 1,
                "created_at": 1,
                "resolution_status": 1,
                "resolved_at": 1,
                "video_status": "$video.status",
                "filename": {
                    "$ifNull": ["$video.original_filename", "$video.stored_filename"],
                },
            }
        },
    ]

    search_text = (video_search or "").strip()
    if search_text:
        escaped = re.escape(search_text)
        pipeline.append({"$match": {"filename": {"$regex": escaped, "$options": "i"}}})

    docs = list(detections.aggregate(pipeline))
    alerts: list[dict[str, Any]] = []
    for doc in docs:
        alert = _serialize_alert(doc)
        if status and status != "all" and status != alert["status"]:
            continue
        alerts.append(alert)

    summary = {
        "total": len(alerts),
        "active": len([a for a in alerts if a["status"] == "active"]),
        "critical": len([a for a in alerts if a["severity"] == "critical"]),
        "resolved": len([a for a in alerts if a["status"] == "resolved"]),
    }

    return {
        "summary": summary,
        "alerts": alerts,
    }


@router.post("/alerts/{alert_id}/resolve")
def resolve_alert(alert_id: str) -> dict[str, Any]:
    _, detections = st.require_db()
    oid = _parse_object_id(alert_id, "Invalid alert id")

    result = detections.update_one(
        {"_id": oid},
        {
            "$set": {
                "resolution_status": "resolved",
                "resolved_at": st.now_utc(),
            }
        },
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Alert not found")

    return {"ok": True, "alert_id": alert_id, "status": "resolved"}


@router.post("/alerts/{alert_id}/toggle-resolve")
def toggle_alert_resolve(alert_id: str) -> dict[str, Any]:
    _, detections = st.require_db()
    oid = _parse_object_id(alert_id, "Invalid alert id")

    doc = detections.find_one({"_id": oid}, {"resolution_status": 1, "resolved_at": 1})
    if not doc:
        raise HTTPException(status_code=404, detail="Alert not found")

    is_resolved = str(doc.get("resolution_status") or "").lower() == "resolved" or bool(doc.get("resolved_at"))

    if is_resolved:
        detections.update_one(
            {"_id": oid},
            {
                "$set": {"resolution_status": "active"},
                "$unset": {"resolved_at": ""},
            },
        )
        next_status = "active"
    else:
        detections.update_one(
            {"_id": oid},
            {
                "$set": {
                    "resolution_status": "resolved",
                    "resolved_at": st.now_utc(),
                }
            },
        )
        next_status = "resolved"

    return {"ok": True, "alert_id": alert_id, "status": next_status}
