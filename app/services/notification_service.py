"""FR-009基盤: Slack Webhook通知サービス"""

import requests


class NotificationService:
    """Slack Webhook通知サービス"""

    def __init__(self, app=None):
        self.webhook_url = None
        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        """Flask拡張パターンの初期化"""
        self.webhook_url = app.config.get("SLACK_WEBHOOK_URL")

    def send(self, message: str, channel: str = None) -> dict:
        """
        WebhookにPOST送信

        Returns:
            dict: {"success": bool, "error": str | None}
        """
        if not self.webhook_url:
            return {"success": False, "error": "SLACK_WEBHOOK_URL未設定"}

        payload = {"text": message}
        if channel:
            payload["channel"] = channel

        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=5,
            )
            response.raise_for_status()
            return {"success": True, "error": None}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    def send_order_status(
        self,
        order_number: str,
        from_status: str,
        to_status: str,
        product_name: str = "",
        error: str = None,
    ) -> dict:
        """自動発注ステータス変更通知"""
        if to_status == "error":
            message = f":rotating_light: 注文 #{order_number} ({product_name}) エラー: {error}"
        elif from_status == to_status:
            message = f":information_source: 注文 #{order_number} ({product_name}) ステータス確認: {to_status}"
        else:
            message = f":package: 注文 #{order_number} ({product_name}): {from_status} → {to_status}"
        return self.send(message)

    def send_pipeline_result(
        self,
        product_name: str,
        status: str,
        elapsed_sec: float,
        errors: str = None,
    ) -> dict:
        """パイプライン結果のフォーマット送信"""
        if status == "success":
            message = f"✅ パイプライン完了: {product_name} ({elapsed_sec}秒)"
        elif status == "partial":
            message = f"⚠️ パイプライン部分成功: {product_name} ({elapsed_sec}秒) - {errors}"
        elif status == "failed":
            message = f"❌ パイプライン失敗: {product_name} - {errors}"
        else:
            message = f"ℹ️ パイプライン結果: {product_name} - ステータス不明 ({status})"

        return self.send(message)
