# warehouse_webhook ルートのテスト
# POST /api/warehouse/events エンドポイントのカバレッジ向上

from __future__ import annotations

import contextlib
import hmac
import json
from hashlib import sha256
from unittest.mock import patch

import pytest

from app import create_app
from app.extensions import db


@pytest.fixture(scope="function")
def app():
    app = create_app()
    app.config.update(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
            "WTF_CSRF_ENABLED": False,
            "SECRET_KEY": "test-secret",
        }
    )
    # warehouse_webhook Blueprint を手動登録（create_app に未登録のため）
    from app.routes.warehouse_webhook import router as warehouse_router

    with contextlib.suppress(ValueError):
        app.register_blueprint(warehouse_router)
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


def _make_signature(body: bytes, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), body, sha256).hexdigest()


class TestForward2meEvents:
    @patch("app.routes.warehouse_webhook.handle_forward2me_event")
    @patch("app.routes.warehouse_webhook._load_forward2me_config")
    def test_valid_signature_returns_ok(self, mock_load_config, mock_handle_event, client):
        secret = "test_webhook_secret"
        mock_load_config.return_value = {"webhook_secret_env": "FWD2ME_SECRET"}
        body_dict = {"event_type": "parcel_received", "parcel_id": "PKG-001"}
        body_bytes = json.dumps(body_dict, separators=(",", ":")).encode("utf-8")
        signature = _make_signature(body_bytes, secret)
        with patch("app.routes.warehouse_webhook.os.getenv", return_value=secret):
            response = client.post(
                "/api/warehouse/events",
                data=body_bytes,
                content_type="application/json",
                headers={"X-Webhook-Signature": signature},
            )
        assert response.status_code == 200
        assert response.get_json() == {"status": "ok"}
        mock_handle_event.assert_called_once()

    @patch("app.routes.warehouse_webhook.handle_forward2me_event")
    @patch("app.routes.warehouse_webhook._load_forward2me_config")
    def test_no_secret_configured_skips_verification(self, mock_load_config, mock_handle_event, client):
        mock_load_config.return_value = {"webhook_secret_env": "FWD2ME_SECRET"}
        with patch("app.routes.warehouse_webhook.os.getenv", return_value=""):
            response = client.post("/api/warehouse/events", json={"event_type": "parcel_received"})
        assert response.status_code == 200
        assert response.get_json() == {"status": "ok"}
        mock_handle_event.assert_called_once()

    @patch("app.routes.warehouse_webhook.handle_forward2me_event")
    @patch("app.routes.warehouse_webhook._load_forward2me_config")
    def test_config_not_found_skips_verification(self, mock_load_config, mock_handle_event, client):
        mock_load_config.return_value = {}
        response = client.post("/api/warehouse/events", json={"event_type": "parcel_received"})
        assert response.status_code == 200
        assert response.get_json() == {"status": "ok"}
        mock_handle_event.assert_called_once()

    @patch("app.routes.warehouse_webhook.handle_forward2me_event")
    @patch("app.routes.warehouse_webhook._load_forward2me_config")
    def test_empty_payload_returns_ok(self, mock_load_config, mock_handle_event, client):
        mock_load_config.return_value = {}
        response = client.post("/api/warehouse/events", data="{}", content_type="application/json")
        assert response.status_code == 200
        assert response.get_json() == {"status": "ok"}
        mock_handle_event.assert_called_once_with({})

    @patch("app.routes.warehouse_webhook.handle_forward2me_event")
    @patch("app.routes.warehouse_webhook._load_forward2me_config")
    def test_invalid_signature_returns_401(self, mock_load_config, mock_handle_event, client):
        mock_load_config.return_value = {"webhook_secret_env": "FWD2ME_SECRET"}
        with patch("app.routes.warehouse_webhook.os.getenv", return_value="correct_secret"):
            response = client.post(
                "/api/warehouse/events",
                json={"event_type": "parcel_received"},
                headers={"X-Webhook-Signature": "invalid_signature_xyz"},
            )
        assert response.status_code == 401
        mock_handle_event.assert_not_called()

    @patch("app.routes.warehouse_webhook.handle_forward2me_event")
    @patch("app.routes.warehouse_webhook._load_forward2me_config")
    def test_missing_signature_header_returns_401(self, mock_load_config, mock_handle_event, client):
        mock_load_config.return_value = {"webhook_secret_env": "FWD2ME_SECRET"}
        with patch("app.routes.warehouse_webhook.os.getenv", return_value="some_secret"):
            response = client.post("/api/warehouse/events", json={"event_type": "parcel_received"})
        assert response.status_code == 401
        mock_handle_event.assert_not_called()


class TestVerifySignature:
    def test_valid_signature(self):
        from app.routes.warehouse_webhook import _verify_signature

        secret = "my_secret"
        body = b'{"event": "test"}'
        signature = hmac.new(secret.encode(), body, sha256).hexdigest()
        assert _verify_signature(body, signature, secret) is True

    def test_wrong_signature(self):
        from app.routes.warehouse_webhook import _verify_signature

        assert _verify_signature(b"body", "wrong_sig", "secret") is False

    def test_empty_signature(self):
        from app.routes.warehouse_webhook import _verify_signature

        assert _verify_signature(b"body", "", "secret") is False


class TestLoadForward2meConfig:
    def test_returns_empty_dict_when_file_not_found(self):
        from app.routes.warehouse_webhook import _load_forward2me_config

        with patch("app.routes.warehouse_webhook.Path.exists", return_value=False):
            result = _load_forward2me_config()
        assert result == {}

    def test_returns_config_when_file_exists(self):
        from app.routes.warehouse_webhook import _load_forward2me_config

        config_data = {"webhook_secret_env": "FWD2ME_SECRET"}
        with (
            patch("app.routes.warehouse_webhook.Path.exists", return_value=True),
            patch("app.routes.warehouse_webhook.Path.read_text", return_value=json.dumps(config_data)),
        ):
            result = _load_forward2me_config()
        assert result == config_data

    def test_returns_empty_dict_on_json_error(self):
        from app.routes.warehouse_webhook import _load_forward2me_config

        with (
            patch("app.routes.warehouse_webhook.Path.exists", return_value=True),
            patch("app.routes.warehouse_webhook.Path.read_text", return_value="not valid json {{{"),
        ):
            result = _load_forward2me_config()
        assert result == {}
