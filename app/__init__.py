# app/__init__.py
import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, Blueprint
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

# --- 拡張 ---
db = SQLAlchemy()
migrate = Migrate()

def create_app(config_name=None):
    """Flaskアプリのファクトリ"""
    app = Flask(__name__, instance_relative_config=True)

    # --- 設定読み込み ---
    # 環境変数 APP_SETTINGS か引数 config_name、無ければ default
    config_name = config_name or os.getenv("APP_SETTINGS", "default")
    try:
        from config import config_map
        app.config.from_object(config_map[config_name])
        # オプションの init_app
        if hasattr(config_map[config_name], "init_app"):
            config_map[config_name].init_app(app)
    except Exception as e:
        app.logger.warning(f"[config] fallback to default due to: {e}")
        from config import Config
        app.config.from_object(Config)

    # インスタンスフォルダ作成（ログ/DBなど保存用）
    try:
        os.makedirs(app.instance_path, exist_ok=True)
    except OSError:
        pass

    # --- ログ設定（回転ログ） ---
    log_dir = os.path.join(app.instance_path, "logs")
    os.makedirs(log_dir, exist_ok=True)
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "app.log"), maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)

    # --- DB / Migrate 初期化 ---
    if not app.config.get("SQLALCHEMY_DATABASE_URI") and not app.config.get("SQLALCHEMY_BINDS"):
        app.logger.error("SQLALCHEMY_DATABASE_URI が未設定です。config.py を確認してください。")

    db.init_app(app)
    migrate.init_app(app, db)

    # --- モデル import（メタデータをMigrateへ） ---
    try:
        from . import models  # noqa: F401
        app.logger.info("Models imported for metadata binding.")
    except Exception as e:
        app.logger.warning(f"Model import failed: {e}")

    # --- Blueprint 登録 ---
    try:
        from .routes import bp as main_bp
        app.register_blueprint(main_bp)
        app.logger.info("Blueprint main_bp registered.")
    except Exception as e:
        app.logger.error(f"Blueprint registration failed: {e}")

        # 最低限のフォールバック（/ -> 200 を返す）
        fallback = Blueprint("fallback", __name__)

        @fallback.route("/")
        def index_fallback():
            return (
                "<h1>Blueprint 未登録のためフォールバック応答</h1>"
                "<p>app/routes.py の Blueprint を確認してください。</p>",
                200,
            )

        app.register_blueprint(fallback)
        app.logger.info("Fallback blueprint registered.")

    app.logger.info("Flask application created successfully.")
    return app

# Flask CLI用
from . import models  # noqa: E402,F401
