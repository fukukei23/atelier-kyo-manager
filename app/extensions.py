# ======================================================================
# プロジェクト : atelier-kyo-manager
# ファイル名   : app/extensions.py   ← 全置換用
# 目的         : Flask 拡張の単一インスタンスを定義（アプリ全体で共有）
# 使い方       :
#   - 他ファイルでは「from app.extensions import db, migrate, csrf」で利用
#   - app/__init__.py の create_app() 内で
#       db.init_app(app); migrate.init_app(app, db); csrf.init_app(app)
#     を呼び出して初期化します。
# メモ         : ここで *新しく* SQLAlchemy() を増やさないこと（1個に統一）
# 日付         : 2025-08-23 (JST)
# ======================================================================

from __future__ import annotations

from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import CSRFProtect

# 単一インスタンス（アプリ全体で共有）
# expire_on_commit=False によりコミット後も属性アクセスが楽になります。
db = SQLAlchemy(session_options={"expire_on_commit": False})

# マイグレーション管理（alembic 連携）
migrate = Migrate()

# CSRF 保護（Flask-WTF）
csrf = CSRFProtect()

__all__ = ["db", "migrate", "csrf", "celery_ext"]

# Celery は app/core/celery_app.py で定義済み。
# Flask連携は create_app() 内で init_celery(app) を呼ぶ。
from app.core.celery_app import celery as celery_ext
