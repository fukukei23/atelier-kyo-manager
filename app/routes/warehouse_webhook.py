from __future__ import annotations

import hmac
import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict

from flask import Blueprint, abort, jsonify, request

from app.services.warehouse_event_service import handle_forward2me_event

router = Blueprint("warehouse_webhook", __name__)


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
    handle_forward2me_event(payload)
    return jsonify({"status": "ok"})


def _verify_signature(body: bytes, signature: str, secret: str) -> bool:
    if not signature:
        return False
    expected = hmac.new(secret.encode("utf-8"), body, sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def _load_forward2me_config() -> Dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "config" / "integrations" / "forward2me.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
