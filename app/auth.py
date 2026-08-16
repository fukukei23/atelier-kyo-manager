# ======================================================================
# プロジェクト: atelier-kyo-manager
# ファイル名  : app/auth.py
# 目的        : 認証 Blueprint（F01: ログイン機能）
# ======================================================================

from __future__ import annotations

from urllib.parse import urlparse

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

bp = Blueprint("auth", __name__)


def _is_safe_next(target: str | None) -> bool:
    """Open Redirect (CWE-601) 対策: nextは自サイト相対パスのみ許可（ISSUE-104）。

    - "/"で始まる相対パス（netloc・schemeを持たない）のみ True
    - 絶対URL・プロトコル相対（//evil.com）・バックスラッシュ変形（/\\evil.com）は False
    - タブ/改行等の制御文字は事前に除去（WHATWG URL仕様でブラウザがURL解析時に
      これらを除去するため、除去前提でチェックしないと "/\t/evil.com" が
      ブラウザ側で実質 "//evil.com" として解釈されるバイパスを許してしまう）
    """
    if not target:
        return False
    target = "".join(c for c in target if c not in "\t\r\n")
    if not target.startswith("/"):
        return False
    if target.startswith("//") or "\\" in target:
        return False
    parsed = urlparse(target)
    return parsed.netloc == "" and parsed.scheme == ""


@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.index"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            flash("ユーザー名とパスワードを入力してください。", "error")
            return render_template("auth/login.html"), 400

        from .models.user import User

        user = User.query.filter_by(username=username).first()

        if user is None or not user.check_password(password):
            flash("ユーザー名またはパスワードが正しくありません。", "error")
            return render_template("auth/login.html"), 401

        if not user.is_active:
            flash("このアカウントは無効化されています。", "error")
            return render_template("auth/login.html"), 403

        login_user(user, remember=True)
        flash(f"ようこそ、{user.display_name or user.username}さん！", "success")

        next_page = request.args.get("next")
        if not _is_safe_next(next_page):
            next_page = None
        return redirect(next_page or url_for("main.index"))

    return render_template("auth/login.html")


@bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("ログアウトしました。", "success")
    return redirect(url_for("auth.login"))
