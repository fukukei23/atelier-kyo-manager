# ======================================================================
# プロジェクト: atelier-kyo-manager
# ファイル名  : app/__init__.py
# 目的        : Flaskアプリの生成（アプリファクトリ）＋ 拡張の初期化
# 重要        : SQLAlchemy/Migrate/CSRF は app/extensions.py の単一インスタンスを使う
# 日付        : 2025-08-23 (JST)
# 備考        : config の場所が「ルート（config.py）」/「app配下」のどちらでも動くよう
#               フォールバック付きのローダーを内蔵
# ======================================================================

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

from flask import Flask

# ★ ここがポイント：自前で SQLAlchemy() を作らない。extensions の単一インスタンスを使う
from app.extensions import csrf, db, migrate


def _load_config(app: Flask, config_name: str | None = None) -> None:
    """
    設定ロードのフォールバック戦略：
      1) 環境変数 APP_CONFIG="module:object" があれば from_object でロード
      2) ルートの config.py に load_config(app) があればそれを実行
      3) ルートの config.py に Config クラスがあれば from_object でロード
      4) app/config.py があれば同様に試行
      5) どれも無ければ最小デフォルト（instance/site.db）
    """
    # 1) APP_CONFIG（例: "config:Config" や "settings:ProdConfig"）
    cfg_path = os.getenv("APP_CONFIG")
    if cfg_path:
        try:
            module, obj = cfg_path.split(":")
            mod = __import__(module, fromlist=[obj])
            app.config.from_object(getattr(mod, obj))
            app.logger.info(f"[config] loaded from APP_CONFIG={cfg_path}")
            return
        except Exception as e:
            app.logger.warning(f"[config] APP_CONFIG load failed: {e}")

    # 2) ルート config.py の load_config(app)
    try:
        import config as root_cfg  # ルート（app/ ではない）

        if hasattr(root_cfg, "load_config"):
            root_cfg.load_config(app)  # すでにあなたのログに出ていた loader を使用
            app.logger.info("[config] loaded via root config.load_config")
            return
    except Exception as e:
        app.logger.warning(f"[config] root load_config not used: {e}")

    # 3) ルート config.py の Config クラス
    try:
        import config as root_cfg

        if hasattr(root_cfg, "Config"):
            app.config.from_object(root_cfg.Config)
            app.logger.info("[config] loaded via root config.Config")
            return
    except Exception as e:
        app.logger.warning(f"[config] root Config not used: {e}")

    # 4) app/config.py 側も一応サーチ（ある環境向け）
    try:
        from . import config as app_cfg  # app/config.py がある場合のみ

        if hasattr(app_cfg, "load_config"):
            app_cfg.load_config(app)
            app.logger.info("[config] loaded via app.config.load_config")
            return
        if hasattr(app_cfg, "Config"):
            app.config.from_object(app_cfg.Config)
            app.logger.info("[config] loaded via app.config.Config")
            return
    except Exception:
        pass

    # 5) フォールバック（最小構成）
    os.makedirs(app.instance_path, exist_ok=True)
    app.config.from_mapping(
        SECRET_KEY=os.getenv("SECRET_KEY", "change-me"),
        SQLALCHEMY_DATABASE_URI=(
            os.getenv("DATABASE_URL") or f"sqlite:///{os.path.join(app.instance_path, 'site.db')}"
        ),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )
    app.logger.warning("[config] fallback minimal config is used.")


def create_app(config_name: str | None = None) -> Flask:
    app = Flask(__name__, instance_relative_config=True)

    # 設定ロード
    _load_config(app, config_name=config_name)

    # ログ
    logs_dir = os.path.join(app.instance_path, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    handler = RotatingFileHandler(os.path.join(logs_dir, "app.log"), maxBytes=1_000_000, backupCount=3)
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)

    # 拡張の初期化（単一インスタンス）
    db.init_app(app)
    migrate.init_app(app, db)
    csrf.init_app(app)

    # モデルは init_app の後で import（順序が重要）
    from app import models  # noqa: F401

    app.logger.info("Models imported for metadata binding.")

    # ルート登録
    from app.routes import bp as main_bp

    app.register_blueprint(main_bp)
    app.logger.info("Blueprint main_bp registered.")

    app.logger.info("Flask application created successfully.")
    return app
