# ======================================================================
# プロジェクト: atelier-kyo-manager
# ファイル名  : app/__init__.py
# 目的        : Flask アプリケーションファクトリ
# 日付        : 2026-04-19 (JST)
# ======================================================================

from __future__ import annotations

from flask import Flask
from flask_login import LoginManager

from .config.config import AppConfig
from .extensions import csrf, db, migrate

__version__ = "2.0.0"

login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.login_message = "ログインが必要です。"


def create_app(config_name: str | None = None) -> Flask:
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    # --- 設定（AppConfig に一元管理） ---
    app.config.update(AppConfig.get_flask_config())

    # --- 拡張初期化 ---
    db.init_app(app)
    migrate.init_app(app, db)
    csrf.init_app(app)
    login_manager.init_app(app)

    # --- Blueprint 登録 ---
    from .routes import bp as main_bp  # noqa: F811

    app.register_blueprint(main_bp)

    from .auth import bp as auth_bp

    app.register_blueprint(auth_bp, url_prefix="/auth")

    # --- モデルをインポート（DB作成用） ---
    with app.app_context():
        from . import models  # noqa: F401

        db.create_all()

    return app


@login_manager.user_loader
def _load_user(user_id: int):
    from .models.user import User

    return db.session.get(User, user_id)
