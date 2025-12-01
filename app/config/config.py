# ==============================================================================
# File: config.py
# Registry: C:\Users\USER\tools\atelier-kyo-manager\app\config\config.py
# Date & Time (JST): 2025-12-01
# Version: 7.0J (NexusCore-style separation)
#
# --- 変更内容 ---
# - 非機密設定のみをConfigクラスに集約（NexusCoreのAppConfig相当）
# - 機密設定（APIキー等）はSecretsクラスから直接取得する方式に変更
# - 設定の責務を明確に分離
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

class SiteSelectors(TypedDict, total=False): pass
class Browser(str, Enum): CHROME = "chrome"; EDGE = "edge"
class LogLevel(str, Enum): DEBUG = "DEBUG"; INFO = "INFO"; WARNING = "WARNING"; ERROR = "ERROR"

def _sqlite_uri(db_path: Path) -> str:
    return f"sqlite:///{db_path.resolve().as_posix()}"

def _load_layered_site_config(config_dir: Path) -> Dict[str, Any]:
    """サイト設定を読み込む（base.jsonから）"""
    base_path = config_dir / "sites" / "base.json"
    if not base_path.exists(): 
        raise FileNotFoundError(f"Base site config not found: {base_path}")
    sites_config = {}
    with base_path.open("r", encoding="utf-8") as f:
        base_data = json.load(f)
        for site in base_data.get("sites", []):
            if "name" in site: 
                sites_config[site["name"]] = site
    return sites_config

class AppConfig:
    """
    アプリ全体の非機密設定（秘密鍵除く）
    環境変数から読み込み、デフォルト値を提供
    
    機密情報（APIキー等）は Secrets クラスから直接取得すること
    """
    # ---- ディレクトリパス（自動生成） ----
    ROOT_DIR: Path = Path(__file__).resolve().parents[2]
    APP_DIR: Path = ROOT_DIR / "app"
    DATA_DIR: Path = ROOT_DIR / "data"
    CONFIG_DIR: Path = APP_DIR / "config"
    
    # ディレクトリを自動作成
    for d in (DATA_DIR, CONFIG_DIR, CONFIG_DIR / "sites"):
        d.mkdir(parents=True, exist_ok=True)

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
    
    # ---- データベース設定（デフォルト値のみ、機密情報はSecretsから） ----
    @staticmethod
    def get_db_url() -> str:
        """データベースURLを取得（Secrets > デフォルト）"""
        try:
            from .secrets import Secrets
            if hasattr(Secrets, "DB_URL") and Secrets.DB_URL:
                return Secrets.DB_URL
        except ImportError:
            pass
        return _sqlite_uri(AppConfig.DATA_DIR / "app.db")

# 後方互換性のため Config を AppConfig のエイリアスとして提供
Config = AppConfig

def setup_logging() -> None:
    """ログ設定を初期化"""
    logging.basicConfig(
        level=AppConfig.LOG_LEVEL.value, 
        format=AppConfig.LOG_FORMAT, 
        stream=sys.stdout
    )
    logging.getLogger("playwright").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

def log_api_keys_visibility() -> None:
    """APIキーの可視性をログに出力（デバッグ用）"""
    logger = logging.getLogger(__name__)
    try:
        from .secrets import Secrets
        def mask(v: Optional[str]) -> str: 
            return f"{v[:4]}***" if v and len(v) > 4 else "Not Set"
        logger.info("--- API Key Diagnostics ---")
        logger.info(f"  GEMINI_API_KEY:   {mask(getattr(Secrets, 'GEMINI_API_KEY', None))}")
        logger.info(f"  DEEPSEEK_API_KEY: {mask(getattr(Secrets, 'DEEPSEEK_API_KEY', None))}")
        logger.info(f"  OPENAI_API_KEY:   {mask(getattr(Secrets, 'OPENAI_API_KEY', None))}")
        logger.info("---------------------------")
    except ImportError:
        logger.warning("secrets.py not found. API keys cannot be checked.")
