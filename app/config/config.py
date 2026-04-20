# ==============================================================================
# File: config.py
# Date: 2026-04-19
# Version: 8.0
#
# --- 変更内容 ---
# - Flask設定（SECRET_KEY, SQLALCHEMY等）を統合
# - 機密情報は .env から os.environ 経由で取得
# - 旧 NexusCore-style ブラウザ設定（SITES, BROWSER等）は維持
# ==============================================================================
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Literal, Optional, TypedDict


class SiteSelectors(TypedDict, total=False):
    pass


class Browser(str, Enum):
    CHROME = "chrome"
    EDGE = "edge"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


def _sqlite_uri(db_path: Path) -> str:
    return f"sqlite:///{db_path.resolve().as_posix()}"


def _load_layered_site_config(config_dir: Path) -> Dict[str, Any]:
    """サイト設定を読み込む（base.jsonから）"""
    base_path = config_dir / "sites" / "base.json"
    if not base_path.exists():
        return {}
    sites_config = {}
    with base_path.open("r", encoding="utf-8") as f:
        base_data = json.load(f)
        for site in base_data.get("sites", []):
            if "name" in site:
                sites_config[site["name"]] = site
    return sites_config


class AppConfig:
    """
    アプリ全体の設定を一元管理するクラス。
    Flask設定 + ブラウザ設定 + サイト設定を統合。

    機密情報は .env ファイルから os.environ 経由で取得。
    """
    # ---- ディレクトリパス ----
    ROOT_DIR: Path = Path(__file__).resolve().parents[2]
    APP_DIR: Path = ROOT_DIR / "app"
    DATA_DIR: Path = ROOT_DIR / "data"
    CONFIG_DIR: Path = APP_DIR / "config"

    # ディレクトリを自動作成
    for d in (DATA_DIR, CONFIG_DIR, CONFIG_DIR / "sites"):
        d.mkdir(parents=True, exist_ok=True)

    # ---- Flask設定 ----
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False

    # ---- ブラウザ設定 ----
    BROWSER: Browser = Browser(os.getenv("AK_BROWSER", "edge").lower())
    HEADLESS: bool = os.getenv("AK_HEADLESS", "false").lower() in {"true", "1", "yes"}

    # ---- ログ設定 ----
    LOG_LEVEL: LogLevel = LogLevel(os.getenv("AK_LOG_LEVEL", "INFO").upper())
    LOG_FORMAT: str = os.getenv("AK_LOG_FORMAT", "[%(asctime)s] %(levelname)-8s | %(message)s")

    # ---- 環境設定 ----
    STAGE: Literal["test", "staging", "prod"] = os.getenv("AK_STAGE", "test")

    # ---- サイト設定 ----
    SITES: Dict[str, Any] = _load_layered_site_config(CONFIG_DIR)

    # ---- データベース設定 ----
    @staticmethod
    def get_db_url() -> str:
        """データベースURLを取得（環境変数 > デフォルトSQLite）"""
        db_url = os.environ.get("DATABASE_URL")
        if db_url:
            return db_url
        return _sqlite_uri(AppConfig.DATA_DIR / "app.db")

    @classmethod
    def get_flask_config(cls) -> Dict[str, Any]:
        """Flask app.config に渡す設定dictを返す"""
        return {
            "SECRET_KEY": cls.SECRET_KEY,
            "SQLALCHEMY_DATABASE_URI": cls.get_db_url(),
            "SQLALCHEMY_TRACK_MODIFICATIONS": cls.SQLALCHEMY_TRACK_MODIFICATIONS,
            "SLACK_WEBHOOK_URL": os.environ.get("SLACK_WEBHOOK_URL", ""),
        }


# 後方互換性のため Config を AppConfig のエイリアスとして提供
Config = AppConfig


def setup_logging() -> None:
    """ログ設定を初期化"""
    logging.basicConfig(
        level=AppConfig.LOG_LEVEL.value,
        format=AppConfig.LOG_FORMAT,
        stream=sys.stdout,
    )
    logging.getLogger("playwright").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
