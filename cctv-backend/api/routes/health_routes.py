from fastapi import APIRouter

from core import app_state as st

router = APIRouter(tags=["health"])


def _mongo_status() -> str:
    return "connected" if st.db_enabled() else "disabled"


@router.get("/health")
def health():
    return {
        "status": "ok",
        "mongodb": _mongo_status(),
        "mongo_error": st.mongo_last_error,
    }


@router.post("/mongo/reconnect")
def mongo_reconnect():
    st.init_mongo()
    return {
        "mongodb": _mongo_status(),
        "mongo_error": st.mongo_last_error,
    }
