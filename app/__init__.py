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
from .services.notification_service import NotificationService

__version__ = "2.0.0"

notification = NotificationService()

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
    notification.init_app(app)
    app.notification = notification

    # --- Blueprint 登録 ---
    from .routes import bp as main_bp  # noqa: F811

    app.register_blueprint(main_bp)

    from .auth import bp as auth_bp

    app.register_blueprint(auth_bp, url_prefix="/auth")

    # --- モデルをインポート（DB作成用） ---
    with app.app_context():
        from . import models  # noqa: F401

        db.create_all()

    # --- テンプレートフィルタ登録 ---
    _register_template_filters(app)

    # --- Celery 初期化 ---
    from .core.celery_app import make_celery

    celery = make_celery(app)
    app.celery = celery

    # --- CLI コマンド登録 ---
    from .commands import register_commands

    register_commands(app)

    return app


def _register_template_filters(app: Flask) -> None:
    from .utils import presentation as pres

    app.jinja_env.filters.update(
        {
            "segment_label": pres.segment_label,
            "segment_color": pres.segment_color,
            "priority_label": pres.priority_label,
            "priority_color": pres.priority_color,
            "partner_status_label": pres.partner_status_label,
            "deadline_color": pres.deadline_color,
            "deadline_message": pres.deadline_message,
            "score_label": pres.score_label,
            "score_color": pres.score_color,
            "shipment_status_label": pres.shipment_status_label,
            "daily_status_color": pres.daily_status_color,
            "monthly_status_color": pres.monthly_status_color,
            "stock_status_label": pres.stock_status_label,
            "stock_status_color": pres.stock_status_color,
            "risk_label": pres.risk_label,
            "risk_color": pres.risk_color,
            "recommendation_label": pres.recommendation_label,
        }
    )


@login_manager.user_loader
def _load_user(user_id: int):
    from .models.user import User

    return db.session.get(User, user_id)
