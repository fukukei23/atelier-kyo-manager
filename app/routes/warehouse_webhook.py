from __future__ import annotations

import hmac
import json
import logging
import os
from hashlib import sha256
from pathlib import Path
from typing import Any

from flask import Blueprint, abort, jsonify, request

from app.services.warehouse_event_service import handle_forward2me_event
from app.utils.validation import sanitize_filename

router = Blueprint("warehouse_webhook", __name__)
logger = logging.getLogger(__name__)


@router.post("/api/warehouse/events")
def forward2me_events():
    cfg = _load_forward2me_config()
    secret_env = cfg.get("webhook_secret_env", "")
    secret = os.getenv(secret_env, "") if secret_env else ""
    body = request.get_data(cache=False)
    signature = request.headers.get("X-Webhook-Signature", "")
    if secret and not _verify_signature(body, signature, secret):
        abort(401, description="Invalid signature")

    payload = request.get_json(force=True, silent=True) or {}

    # parcel_idサニタイズ
    if "parcel_id" in payload:
        payload["parcel_id"] = sanitize_filename(str(payload["parcel_id"]))

    try:
        handle_forward2me_event(payload)
    except Exception as e:
        logger.error("Webhook handler error: %s", e, exc_info=True)
        # エラー時も200を返す（Webhookリトライ防止）
        return jsonify({"status": "error", "message": "processing failed"})

    return jsonify({"status": "ok"})


def _verify_signature(body: bytes, signature: str, secret: str) -> bool:
    if not signature:
        return False
    expected = hmac.new(secret.encode("utf-8"), body, sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def _load_forward2me_config() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "config" / "integrations" / "forward2me.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
