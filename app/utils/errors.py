from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def safe_error_msg(error: Exception, *, context: str = "") -> str:
    """STAGE環境に応じてエラーメッセージを切り替える。

    test/staging: 元の例外メッセージ（デバッグ用）
    prod: 汎用メッセージ（情報漏洩防止）
    """
    from flask import current_app

    stage = current_app.config.get("STAGE", "test")

    if stage == "prod":
        logger.error("Error in %s: %s", context, error, exc_info=True)
        return "処理中にエラーが発生しました。しばらくしてからお試しください。"

    detail = f"{context}: " if context else ""
    return f"{detail}{error}"
