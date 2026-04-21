from datetime import timedelta
from typing import Any

from fastapi import APIRouter

from core import app_state as st

router = APIRouter(tags=["analytics"])


def _clamp_days(days: int) -> int:
    return max(1, min(days, 90))


def _trend_change_percent(current: int, previous: int) -> float:
    if previous == 0 and current > 0:
        return 100.0
    if previous == 0:
        return 0.0
    return ((current - previous) / previous) * 100.0


def _time_pattern_info(count: int) -> tuple[str, str]:
    if count >= 15:
        return "Critical", "Peak anomaly window"
    if count >= 8:
        return "High", "Elevated anomaly activity"
    if count >= 4:
        return "Medium", "Moderate activity"
    return "Low", "Minimal activity"


@router.get("/analytics/summary")
def analytics_summary(days: int = 7) -> dict[str, Any]:
    videos, detections = st.require_db()

    days = _clamp_days(days)
    start = st.now_utc() - timedelta(days=days - 1)
    prev_start = start - timedelta(days=days)

    def aggregate_counts(match_filter: dict[str, Any]) -> dict[str, int]:
        rows = list(
            detections.aggregate(
                [
                    {"$match": match_filter},
                    {"$group": {"_id": "$anomaly_id", "count": {"$sum": 1}}},
                ]
            )
        )
        return {str(r["_id"]): int(r["count"]) for r in rows}

    current_counts = aggregate_counts({"created_at": {"$gte": start}})
    previous_counts = aggregate_counts({"created_at": {"$gte": prev_start, "$lt": start}})

    anomaly_trends = []
    all_keys = sorted(set(current_counts) | set(previous_counts))
    for key in all_keys:
        current = current_counts.get(key, 0)
        previous = previous_counts.get(key, 0)
        change_pct = _trend_change_percent(current, previous)

        anomaly_trends.append(
            {
                "anomaly_id": key,
                "current": current,
                "previous": previous,
                "trend": "up" if change_pct >= 0 else "down",
                "change": round(change_pct, 1),
            }
        )

    hour_rows = list(
        detections.aggregate(
            [
                {"$match": {"created_at": {"$gte": start}}},
                {"$group": {"_id": {"$hour": "$created_at"}, "count": {"$sum": 1}}},
            ]
        )
    )
    hour_counts = {int(r["_id"]): int(r["count"]) for r in hour_rows}

    slots = [
        ("00:00-06:00", 0, 6),
        ("06:00-12:00", 6, 12),
        ("12:00-18:00", 12, 18),
        ("18:00-24:00", 18, 24),
    ]
    time_patterns = []
    for label, start_h, end_h in slots:
        count = sum(hour_counts.get(h, 0) for h in range(start_h, end_h))
        severity, pattern = _time_pattern_info(count)

        time_patterns.append(
            {
                "time": label,
                "anomalies": count,
                "severity": severity,
                "pattern": pattern,
            }
        )

    top_videos = list(
        videos.find(
            {"created_at": {"$gte": start}},
            {"original_filename": 1, "total_detections": 1, "status": 1},
        )
        .sort("total_detections", -1)
        .limit(5)
    )

    hot_zones = []
    for row in top_videos:
        total = int(row.get("total_detections", 0) or 0)
        intensity = min(100, total * 8)
        risk = "High" if total >= 10 else "Medium" if total >= 5 else "Low"
        hot_zones.append(
            {
                "zone": str(row.get("original_filename") or "Unknown source"),
                "anomalies": total,
                "risk": risk,
                "intensity": intensity,
            }
        )

    return {
        "days": days,
        "anomaly_trends": anomaly_trends,
        "time_patterns": time_patterns,
        "hot_zones": hot_zones,
    }
