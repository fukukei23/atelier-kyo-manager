# ======================================================================
# プロジェクト: atelier-kyo-manager
# ファイル名  : app/auth.py
# 目的        : 認証 Blueprint（F01: ログイン機能）
# ======================================================================

from __future__ import annotations

from flask import Blueprint, redirect, render_template, request, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user

from .extensions import db

bp = Blueprint("auth", __name__)


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
        return redirect(next_page or url_for("main.index"))

    return render_template("auth/login.html")


@bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("ログアウトしました。", "success")
    return redirect(url_for("auth.login"))
