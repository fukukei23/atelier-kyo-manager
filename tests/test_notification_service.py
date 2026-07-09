"""NotificationService のテスト — ユニットテスト（mock使用）"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.services.notification_service import NotificationService


class TestInit:
    def test_init_without_app(self):
        svc = NotificationService()
        assert svc.webhook_url is None

    def test_init_with_app(self):
        app = MagicMock()
        app.config.get.return_value = "https://hooks.slack.com/test"
        svc = NotificationService(app)
        assert svc.webhook_url == "https://hooks.slack.com/test"


class TestInitApp:
    def test_init_app_sets_webhook(self):
        app = MagicMock()
        app.config.get.return_value = "https://hooks.slack.com/test"
        svc = NotificationService()
        svc.init_app(app)
        assert svc.webhook_url == "https://hooks.slack.com/test"


class TestSend:
    def test_send_no_webhook_url(self):
        svc = NotificationService()
        result = svc.send("hello")
        assert result["success"] is False
        assert "未設定" in result["error"]

    @patch("app.services.notification_service.requests")
    def test_send_success(self, mock_requests):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_requests.post.return_value = mock_response
        mock_requests.exceptions.RequestException = Exception

        svc = NotificationService()
        svc.webhook_url = "https://hooks.slack.com/test"
        result = svc.send("hello")
        assert result["success"] is True

    @patch("app.services.notification_service.requests")
    def test_send_with_channel(self, mock_requests):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_requests.post.return_value = mock_response
        mock_requests.exceptions.RequestException = Exception

        svc = NotificationService()
        svc.webhook_url = "https://hooks.slack.com/test"
        svc.send("hello", channel="#alerts")

        call_args = mock_requests.post.call_args
        assert call_args[1]["json"]["channel"] == "#alerts"

    @patch("app.services.notification_service.requests")
    def test_send_request_exception(self, mock_requests):
        mock_requests.exceptions.RequestException = Exception
        mock_requests.post.side_effect = Exception("connection failed")

        svc = NotificationService()
        svc.webhook_url = "https://hooks.slack.com/test"
        result = svc.send("hello")
        assert result["success"] is False
        assert "connection failed" in result["error"]


class TestSendOrderStatus:
    @patch("app.services.notification_service.requests")
    def test_error_status(self, mock_requests):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_requests.post.return_value = mock_response
        mock_requests.exceptions.RequestException = Exception

        svc = NotificationService()
        svc.webhook_url = "https://hooks.slack.com/test"
        result = svc.send_order_status("ORD001", "pending", "error", "Test Product", "timeout")
        assert result["success"] is True
        call_msg = mock_requests.post.call_args[1]["json"]["text"]
        assert "エラー" in call_msg

    @patch("app.services.notification_service.requests")
    def test_same_status(self, mock_requests):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_requests.post.return_value = mock_response
        mock_requests.exceptions.RequestException = Exception

        svc = NotificationService()
        svc.webhook_url = "https://hooks.slack.com/test"
        result = svc.send_order_status("ORD001", "shipped", "shipped", "Test Product")
        assert result["success"] is True
        call_msg = mock_requests.post.call_args[1]["json"]["text"]
        assert "ステータス確認" in call_msg

    @patch("app.services.notification_service.requests")
    def test_normal_status_change(self, mock_requests):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_requests.post.return_value = mock_response
        mock_requests.exceptions.RequestException = Exception

        svc = NotificationService()
        svc.webhook_url = "https://hooks.slack.com/test"
        result = svc.send_order_status("ORD001", "pending", "shipped", "Test Product")
        assert result["success"] is True
        call_msg = mock_requests.post.call_args[1]["json"]["text"]
        assert "pending" in call_msg and "shipped" in call_msg


class TestSendPipelineResult:
    @patch("app.services.notification_service.requests")
    def test_success_status(self, mock_requests):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_requests.post.return_value = mock_response
        mock_requests.exceptions.RequestException = Exception

        svc = NotificationService()
        svc.webhook_url = "https://hooks.slack.com/test"
        svc.send_pipeline_result("Test Product", "success", 5.2)
        call_msg = mock_requests.post.call_args[1]["json"]["text"]
        assert "完了" in call_msg

    @patch("app.services.notification_service.requests")
    def test_partial_status(self, mock_requests):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_requests.post.return_value = mock_response
        mock_requests.exceptions.RequestException = Exception

        svc = NotificationService()
        svc.webhook_url = "https://hooks.slack.com/test"
        svc.send_pipeline_result("Test Product", "partial", 3.1, errors="image failed")
        call_msg = mock_requests.post.call_args[1]["json"]["text"]
        assert "部分成功" in call_msg

    @patch("app.services.notification_service.requests")
    def test_failed_status(self, mock_requests):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_requests.post.return_value = mock_response
        mock_requests.exceptions.RequestException = Exception

        svc = NotificationService()
        svc.webhook_url = "https://hooks.slack.com/test"
        svc.send_pipeline_result("Test Product", "failed", 0, errors="crash")
        call_msg = mock_requests.post.call_args[1]["json"]["text"]
        assert "失敗" in call_msg

    @patch("app.services.notification_service.requests")
    def test_unknown_status(self, mock_requests):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_requests.post.return_value = mock_response
        mock_requests.exceptions.RequestException = Exception

        svc = NotificationService()
        svc.webhook_url = "https://hooks.slack.com/test"
        svc.send_pipeline_result("Test Product", "weird", 1.0)
        call_msg = mock_requests.post.call_args[1]["json"]["text"]
        assert "不明" in call_msg
