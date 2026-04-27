"""
notification_service.py テスト (Issue #73)

NotificationService の全メソッドをカバー
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from app.services.notification_service import NotificationService


@pytest.fixture
def service():
    """webhook_url 設定済み"""
    svc = NotificationService()
    svc.webhook_url = "https://hooks.slack.com/test"
    return svc


@pytest.fixture
def service_no_url():
    """webhook_url 未設定"""
    return NotificationService()


# === __init__ / init_app ===


def test_init_with_app():
    """app 渡しで init_app が呼ばれる"""
    app = MagicMock()
    app.config = {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/init"}
    svc = NotificationService(app)
    assert svc.webhook_url == "https://hooks.slack.com/init"


def test_init_without_app():
    """app なし → webhook_url is None"""
    svc = NotificationService()
    assert svc.webhook_url is None


def test_init_app():
    """init_app で webhook_url を設定"""
    app = MagicMock()
    app.config = {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/late"}
    svc = NotificationService()
    svc.init_app(app)
    assert svc.webhook_url == "https://hooks.slack.com/late"


# === send ===


def test_send_success(service):
    """正常送信"""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    with patch("app.services.notification_service.requests.post", return_value=mock_resp):
        result = service.send("hello")
    assert result == {"success": True, "error": None}


def test_send_with_channel(service):
    """channel 指定あり"""
    mock_resp = MagicMock()
    with patch("app.services.notification_service.requests.post", return_value=mock_resp) as mock_post:
        service.send("hello", channel="#general")
    call_kwargs = mock_post.call_args
    assert call_kwargs[1]["json"]["channel"] == "#general"


def test_send_no_webhook(service_no_url):
    """webhook_url 未設定 → 失敗"""
    result = service_no_url.send("hello")
    assert result["success"] is False
    assert "未設定" in result["error"]


def test_send_request_exception(service):
    """requests 例外 → 失敗"""
    with patch(
        "app.services.notification_service.requests.post",
        side_effect=requests.exceptions.ConnectionError("timeout"),
    ):
        result = service.send("hello")
    assert result["success"] is False
    assert "timeout" in result["error"]


def test_send_http_error(service):
    """HTTP ステータスエラー → 失敗"""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500")
    with patch("app.services.notification_service.requests.post", return_value=mock_resp):
        result = service.send("hello")
    assert result["success"] is False


# === send_order_status ===


def test_order_status_error(service):
    """to_status=error → エラーメッセージ"""
    with patch("app.services.notification_service.requests.post") as mock_post:
        mock_post.return_value = MagicMock()
        result = service.send_order_status("ORD-1", "pending", "error", "Widget", "在庫なし")
    assert result["success"] is True
    sent = mock_post.call_args[1]["json"]["text"]
    assert ":rotating_light:" in sent
    assert "在庫なし" in sent


def test_order_status_same(service):
    """from_status == to_status → 確認メッセージ"""
    with patch("app.services.notification_service.requests.post") as mock_post:
        mock_post.return_value = MagicMock()
        result = service.send_order_status("ORD-2", "shipped", "shipped", "Gadget")
    sent = mock_post.call_args[1]["json"]["text"]
    assert ":information_source:" in sent
    assert "確認" in sent


def test_order_status_transition(service):
    """正常ステータス遷移"""
    with patch("app.services.notification_service.requests.post") as mock_post:
        mock_post.return_value = MagicMock()
        result = service.send_order_status("ORD-3", "pending", "shipped", "Thing")
    sent = mock_post.call_args[1]["json"]["text"]
    assert ":package:" in sent
    assert "pending" in sent
    assert "shipped" in sent


# === send_pipeline_result ===


def test_pipeline_success(service):
    """status=success"""
    with patch("app.services.notification_service.requests.post") as mock_post:
        mock_post.return_value = MagicMock()
        result = service.send_pipeline_result("ProductA", "success", 12.5)
    sent = mock_post.call_args[1]["json"]["text"]
    assert "✅" in sent
    assert "12.5" in sent


def test_pipeline_partial(service):
    """status=partial"""
    with patch("app.services.notification_service.requests.post") as mock_post:
        mock_post.return_value = MagicMock()
        result = service.send_pipeline_result("ProductB", "partial", 8.0, "画像欠落")
    sent = mock_post.call_args[1]["json"]["text"]
    assert "⚠️" in sent
    assert "画像欠落" in sent


def test_pipeline_failed(service):
    """status=failed"""
    with patch("app.services.notification_service.requests.post") as mock_post:
        mock_post.return_value = MagicMock()
        result = service.send_pipeline_result("ProductC", "failed", 0, "タイムアウト")
    sent = mock_post.call_args[1]["json"]["text"]
    assert "❌" in sent
    assert "タイムアウト" in sent


def test_pipeline_unknown(service):
    """status=unknown"""
    with patch("app.services.notification_service.requests.post") as mock_post:
        mock_post.return_value = MagicMock()
        result = service.send_pipeline_result("ProductD", "weird", 1.0)
    sent = mock_post.call_args[1]["json"]["text"]
    assert "ℹ️" in sent
    assert "weird" in sent
