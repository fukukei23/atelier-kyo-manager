from __future__ import annotations

import functools
import logging

from flask import flash, jsonify, redirect, request, url_for

from app.extensions import db

logger = logging.getLogger(__name__)


def handle_db_error(fallback_endpoint: str | None = None):
    """DB操作のロールバック + フラッシュメッセージを処理するデコレータ。

    Args:
        fallback_endpoint: エラー時のリダイレクト先エンドポイント名。
            Noneの場合はリファラまたは main.index にフォールバック。
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                db.session.rollback()
                logger.warning("DB operation failed in %s: %s", func.__name__, e)
                flash(f"操作に失敗しました: {e}", "error")
                redirect_url = (
                    url_for(fallback_endpoint) if fallback_endpoint else request.referrer or url_for("main.index")
                )
                return redirect(redirect_url)

        return wrapper

    return decorator


def handle_api_error(error_key: str = "error", status_code: int = 500):
    """APIエンドポイント用のDBロールバック + JSONエラーレスポンスデコレータ。

    Args:
        error_key: JSONレスポンスのエラーキー名。デフォルト "error"。
        status_code: エラー時のHTTPステータスコード。デフォルト 500。
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                db.session.rollback()
                logger.warning("API operation failed in %s: %s", func.__name__, e)
                return jsonify({"success": False, error_key: str(e)}), status_code

        return wrapper

    return decorator
