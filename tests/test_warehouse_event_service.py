"""
warehouse_event_service のテスト — Webhook処理のビジネスロジック検証
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


class TestHandleForward2meEvent:
    """handle_forward2me_event のルーティング検証"""

    @patch("app.services.warehouse_event_service._handle_parcel_received")
    def test_routes_parcel_received(self, mock_handler):
        from app.services.warehouse_event_service import handle_forward2me_event

        handle_forward2me_event({"event_type": "parcel_received", "parcel_id": "P001"})
        mock_handler.assert_called_once()

    @patch("app.services.warehouse_event_service._handle_parcel_received")
    def test_ignores_unknown_event_type(self, mock_handler):
        from app.services.warehouse_event_service import handle_forward2me_event

        handle_forward2me_event({"event_type": "unknown_event"})
        mock_handler.assert_not_called()

    @patch("app.services.warehouse_event_service._handle_parcel_received")
    def test_ignores_missing_event_type(self, mock_handler):
        from app.services.warehouse_event_service import handle_forward2me_event

        handle_forward2me_event({})
        mock_handler.assert_not_called()


class TestHandleParcelReceived:
    """_handle_parcel_received のビジネスロジック検証"""

    @patch("app.services.warehouse_event_service._log_reporting_event")
    @patch("app.services.warehouse_event_service._persist_event")
    @patch("app.services.warehouse_event_service.AIVisionAgent")
    @patch("app.services.warehouse_event_service._resolve_run_context")
    def test_persists_inspection_result(self, mock_resolve, mock_vision_cls, mock_persist, mock_report):
        from app.services.warehouse_event_service import _handle_parcel_received

        mock_resolve.return_value = None
        mock_agent = MagicMock()
        mock_agent.inspect_parcel_images.return_value = {"authentic": True}
        mock_vision_cls.return_value = mock_agent

        _handle_parcel_received(
            {
                "parcel_id": "P001",
                "client_reference": "REF-001",
                "photos": ["https://img.example.com/1.jpg"],
                "received_at": "2026-05-10T00:00:00Z",
            }
        )

        mock_persist.assert_called_once()
        record = mock_persist.call_args[0][1]
        assert record["event_type"] == "parcel_received"
        assert record["parcel_id"] == "P001"
        assert record["inspection"] == {"authentic": True}

    @patch("app.services.warehouse_event_service._log_reporting_event")
    @patch("app.services.warehouse_event_service._persist_event")
    @patch("app.services.warehouse_event_service.AIVisionAgent")
    @patch("app.services.warehouse_event_service._resolve_run_context")
    def test_updates_run_context_status(self, mock_resolve, mock_vision_cls, mock_persist, mock_report):
        from app.services.warehouse_event_service import _handle_parcel_received

        mock_run_ctx = MagicMock()
        mock_resolve.return_value = mock_run_ctx
        mock_vision_cls.return_value = MagicMock()

        _handle_parcel_received({"parcel_id": "P001", "client_reference": "REF-001"})

        mock_run_ctx.update_status.assert_called_once_with("WAREHOUSE_RECEIVED", meta={"parcel_id": "P001"})

    @patch("app.services.warehouse_event_service._log_reporting_event")
    @patch("app.services.warehouse_event_service._persist_event")
    @patch("app.services.warehouse_event_service.AIVisionAgent")
    @patch("app.services.warehouse_event_service._resolve_run_context")
    def test_run_context_failure_does_not_crash(self, mock_resolve, mock_vision_cls, mock_persist, mock_report):
        from app.services.warehouse_event_service import _handle_parcel_received

        mock_run_ctx = MagicMock()
        mock_run_ctx.update_status.side_effect = RuntimeError("DB error")
        mock_resolve.return_value = mock_run_ctx
        mock_vision_cls.return_value = MagicMock()

        # Should not raise
        _handle_parcel_received({"parcel_id": "P001", "client_reference": "REF-001"})

    @patch("app.services.warehouse_event_service._log_reporting_event")
    @patch("app.services.warehouse_event_service._persist_event")
    @patch("app.services.warehouse_event_service.AIVisionAgent")
    @patch("app.services.warehouse_event_service._resolve_run_context")
    def test_no_parcel_id_returns_early(self, mock_resolve, mock_vision_cls, mock_persist, mock_report):
        from app.services.warehouse_event_service import _handle_parcel_received

        _handle_parcel_received({"parcel_id": None, "client_reference": "REF-001"})

        mock_persist.assert_not_called()
        mock_vision_cls.assert_not_called()

    @patch("app.services.warehouse_event_service._log_reporting_event")
    @patch("app.services.warehouse_event_service._persist_event")
    @patch("app.services.warehouse_event_service.AIVisionAgent")
    @patch("app.services.warehouse_event_service._resolve_run_context")
    def test_empty_photos_list(self, mock_resolve, mock_vision_cls, mock_persist, mock_report):
        from app.services.warehouse_event_service import _handle_parcel_received

        mock_resolve.return_value = None
        mock_vision_cls.return_value = MagicMock()

        _handle_parcel_received({"parcel_id": "P001", "photos": None})

        record = mock_persist.call_args[0][1]
        assert record["photos"] == []


class TestResolveRunContext:
    """_resolve_run_context の分岐検証"""

    @patch("app.services.warehouse_event_service.RunContext", None)
    def test_returns_none_when_no_client_reference(self):
        from app.services.warehouse_event_service import _resolve_run_context

        assert _resolve_run_context(None) is None
        assert _resolve_run_context("") is None

    @patch("app.services.warehouse_event_service.RunContext", None)
    def test_returns_none_when_run_context_not_available(self):
        from app.services.warehouse_event_service import _resolve_run_context

        assert _resolve_run_context("REF-001") is None

    @patch("app.services.warehouse_event_service.RunContext")
    def test_returns_resolved_context(self, mock_cls):
        from app.services.warehouse_event_service import _resolve_run_context

        mock_ctx = MagicMock()
        mock_cls.load_by_client_reference.return_value = mock_ctx

        result = _resolve_run_context("REF-001")
        assert result == mock_ctx

    @patch("app.services.warehouse_event_service.RunContext")
    def test_returns_none_on_lookup_failure(self, mock_cls):
        from app.services.warehouse_event_service import _resolve_run_context

        mock_cls.load_by_client_reference.side_effect = ValueError("not found")

        result = _resolve_run_context("REF-INVALID")
        assert result is None

    @patch("app.services.warehouse_event_service.RunContext")
    def test_returns_none_when_no_load_method(self, mock_cls):
        from app.services.warehouse_event_service import _resolve_run_context

        # load_by_client_reference メソッドなし
        del mock_cls.load_by_client_reference

        result = _resolve_run_context("REF-001")
        assert result is None


class TestPersistEvent:
    """_persist_event のファイル書き込み検証"""

    @patch("app.services.warehouse_event_service.EVENT_LOG_DIR")
    def test_writes_json_file(self, mock_dir, tmp_path):
        from app.services.warehouse_event_service import _persist_event

        mock_dir.__truediv__ = lambda self, key: tmp_path / key
        mock_dir.mkdir = MagicMock()

        record = {"event_type": "parcel_received", "parcel_id": "P001"}

        with patch("app.services.warehouse_event_service.EVENT_LOG_DIR", tmp_path):
            _persist_event("P001", record)

        files = list(tmp_path.glob("P001_*.json"))
        assert len(files) == 1
        saved = json.loads(files[0].read_text(encoding="utf-8"))
        assert saved["parcel_id"] == "P001"


class TestLogReportingEvent:
    """_log_reporting_event の分岐検証"""

    @patch("app.services.warehouse_event_service.REPORTING_AGENT", None)
    def test_noop_when_no_agent(self):
        from app.services.warehouse_event_service import _log_reporting_event

        # Should not raise
        _log_reporting_event({"test": True})

    @patch("app.services.warehouse_event_service.REPORTING_AGENT")
    def test_calls_agent(self, mock_agent):
        from app.services.warehouse_event_service import _log_reporting_event

        record = {"event_type": "parcel_received"}
        _log_reporting_event(record)
        mock_agent.log_warehouse_event.assert_called_once_with(record)

    @patch("app.services.warehouse_event_service.REPORTING_AGENT")
    def test_agent_failure_does_not_crash(self, mock_agent):
        from app.services.warehouse_event_service import _log_reporting_event

        mock_agent.log_warehouse_event.side_effect = RuntimeError("fail")

        # Should not raise
        _log_reporting_event({"test": True})
