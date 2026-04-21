from datetime import timedelta
from typing import Any

from fastapi import APIRouter

from core import app_state as st

router = APIRouter(tags=["stats"])


def _clamp_days(days: int) -> int:
    return max(1, min(days, 90))


def _daily_counts(collection, start) -> list[dict[str, Any]]:
    rows = list(
        collection.aggregate(
            [
                {"$match": {"created_at": {"$gte": start}}},
                {
                    "$group": {
                        "_id": {
                            "$dateToString": {
                                "format": "%Y-%m-%d",
                                "date": "$created_at",
                            }
                        },
                        "count": {"$sum": 1},
                    }
                },
                {"$sort": {"_id": 1}},
            ]
        )
    )
    return [{"date": row["_id"], "count": row["count"]} for row in rows]


@router.get("/stats/overview")
def stats_overview() -> dict[str, Any]:
    videos, detections = st.require_db()

    total_videos = videos.count_documents({})
    processing_videos = videos.count_documents({"status": "processing"})
    completed_videos = videos.count_documents({"status": "completed"})
    failed_videos = videos.count_documents({"status": "failed"})
    total_detections = detections.count_documents({})

    anomaly_breakdown = list(
        detections.aggregate(
            [
                {"$group": {"_id": "$anomaly_id", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
            ]
        )
    )

    recent_videos = list(
        videos.find(
            {},
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
        .sort("created_at", -1)
        .limit(10)
    )

    for rv in recent_videos:
        rv["id"] = str(rv.pop("_id"))

    return {
        "total_videos": total_videos,
        "processing_videos": processing_videos,
        "completed_videos": completed_videos,
        "failed_videos": failed_videos,
        "total_detections": total_detections,
        "anomaly_breakdown": [
            {"anomaly_id": row["_id"], "count": row["count"]} for row in anomaly_breakdown
        ],
        "recent_videos": recent_videos,
    }


@router.get("/stats/trends")
def stats_trends(days: int = 7) -> dict[str, Any]:
    videos, detections = st.require_db()

    days = _clamp_days(days)
    start = st.now_utc() - timedelta(days=days - 1)

    videos_by_day = _daily_counts(videos, start)
    detections_by_day = _daily_counts(detections, start)

    anomaly_trends = list(
        detections.aggregate(
            [
                {"$match": {"created_at": {"$gte": start}}},
                {
                    "$group": {
                        "_id": {
                            "date": {
                                "$dateToString": {
                                    "format": "%Y-%m-%d",
                                    "date": "$created_at",
                                }
                            },
                            "anomaly_id": "$anomaly_id",
                        },
                        "count": {"$sum": 1},
                    }
                },
                {"$sort": {"_id.date": 1}},
            ]
        )
    )

    return {
        "days": days,
        "videos_by_day": videos_by_day,
        "detections_by_day": detections_by_day,
        "anomaly_trends": [
            {
                "date": row["_id"]["date"],
                "anomaly_id": row["_id"]["anomaly_id"],
                "count": row["count"],
            }
            for row in anomaly_trends
        ],
    }
