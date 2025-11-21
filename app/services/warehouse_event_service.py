from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from app.core.run_context import RunContext
except Exception:  # pragma: no cover - optional import guard
    RunContext = None  # type: ignore

from app.agents.ai_vision_agent import AIVisionAgent
try:
    from app.agents.reporting_agent import ReportingAgent
except Exception:
    ReportingAgent = None  # type: ignore

logger = logging.getLogger(__name__)

EVENT_LOG_DIR = Path("output/warehouse_events")
EVENT_LOG_DIR.mkdir(parents=True, exist_ok=True)
REPORTING_AGENT = ReportingAgent() if ReportingAgent else None


def handle_forward2me_event(payload: Dict[str, Any]) -> None:
    event_type = payload.get("event_type")
    if event_type == "parcel_received":
        _handle_parcel_received(payload)
    else:
        logger.info("[WarehouseEvent] Unsupported event_type=%s", event_type)


def _handle_parcel_received(payload: Dict[str, Any]) -> None:
    parcel_id = payload.get("parcel_id")
    client_reference = payload.get("client_reference")
    photos: List[str] = payload.get("photos") or []
    if not parcel_id:
        logger.warning("[WarehouseEvent] parcel_received missing parcel_id")
        return

    run_ctx = _resolve_run_context(client_reference)
    vision_agent = AIVisionAgent(run_ctx)
    inspection_result = vision_agent.inspect_parcel_images(
        parcel_id=parcel_id,
        image_urls=photos,
    )

    record = {
        "event_type": "parcel_received",
        "parcel_id": parcel_id,
        "client_reference": client_reference,
        "received_at": payload.get("received_at"),
        "photos": photos,
        "inspection": inspection_result,
        "raw_event": payload,
    }
    _persist_event(parcel_id, record)
    _log_reporting_event(record)

    if run_ctx:
        try:
            run_ctx.update_status("WAREHOUSE_RECEIVED", meta={"parcel_id": parcel_id})
        except Exception as exc:
            logger.warning("[WarehouseEvent] Failed to update status: %s", exc)


def _resolve_run_context(client_reference: Optional[str]):
    if not client_reference or RunContext is None:
        return None
    resolver = getattr(RunContext, "load_by_client_reference", None)
    if callable(resolver):
        try:
            return resolver(client_reference)
        except Exception as exc:
            logger.warning(
                "[WarehouseEvent] RunContext lookup failed (client_reference=%s): %s",
                client_reference,
                exc,
            )
    return None


def _persist_event(parcel_id: str, record: Dict[str, Any]) -> None:
    ts = int(time.time())
    path = EVENT_LOG_DIR / f"{parcel_id}_{ts}.json"
    try:
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("[WarehouseEvent] Recorded event -> %s", path)
    except Exception as exc:
        logger.error("[WarehouseEvent] Failed to persist event: %s", exc)


def _log_reporting_event(record: Dict[str, Any]) -> None:
    if REPORTING_AGENT is None:
        return
    try:
        REPORTING_AGENT.log_warehouse_event(record)
    except Exception as exc:
        logger.warning("[WarehouseEvent] ReportingAgent logging failed: %s", exc)
