from __future__ import annotations

from collections.abc import Callable
from functools import wraps

from flask import abort, flash, redirect, request, url_for
from flask_login import current_user

from .extensions import db


def handle_db(
    success_msg: str = "保存しました",
    error_msg: str | None = None,
    redirect_endpoint: str | None = None,
) -> Callable:
    """
    DB操作を行うビュー関数に例外処理を付与するデコレータ。

    - 正常終了: 関数の戻り値をそのまま返す
    - 例外発生: rollback → flash → redirect
    """

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                db.session.rollback()
                flash(error_msg or f"エラー: {e}", "error")
                if redirect_endpoint:
                    return redirect(url_for(redirect_endpoint))
                return redirect(request.url)

        return wrapper

    return decorator


def admin_required(f: Callable) -> Callable:
    """
    管理者権限が必要なビューに適用。
    @login_required の後に使用すること。
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        if not (current_user.is_authenticated and current_user.is_admin):
            abort(403)
        return f(*args, **kwargs)

    return wrapper
