# ==============================================================================
# File: config.py
# Registry: C:\Users\USER\tools\atelier-kyo-manager\app\config\config.py
# Date & Time (JST): 2025-09-17 11:38:00
# Version: 6.1J (Headful-First Policy)
#
# --- Operation Policy ---
# 当面はヘッドフル運用を標準とする。--slow-mo を併用し、UIの状態変化を必ず
# 目視確認のうえセレクタを更新する。実運用移行時にヘッドレス切替を検討。
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

try:
    from .secrets import Secrets
except ImportError:
    raise RuntimeError("secrets.py not found. Please run 'python app/config/generate_secrets.py'")

class SiteSelectors(TypedDict, total=False): pass
class Browser(str, Enum): CHROME = "chrome"; EDGE = "edge"
class LogLevel(str, Enum): DEBUG = "DEBUG"; INFO = "INFO"; WARNING = "WARNING"; ERROR = "ERROR"

def _sqlite_uri(db_path: Path) -> str:
    return f"sqlite:///{db_path.resolve().as_posix()}"

def _load_layered_site_config(config_dir: Path) -> Dict[str, Any]:
    # This function remains unchanged
    base_path = config_dir / "sites" / "base.json"
    if not base_path.exists(): raise FileNotFoundError(f"Base site config not found: {base_path}")
    sites_config = {}
    with base_path.open("r", encoding="utf-8") as f:
        base_data = json.load(f)
        for site in base_data.get("sites", []):
            if "name" in site: sites_config[site["name"]] = site
    return sites_config

class Config:
    ROOT_DIR: Path = Path(__file__).resolve().parents[2]
    APP_DIR: Path = ROOT_DIR / "app"
    DATA_DIR: Path = ROOT_DIR / "data"
    CONFIG_DIR: Path = APP_DIR / "config"
    for d in (DATA_DIR, CONFIG_DIR, CONFIG_DIR / "sites"):
        d.mkdir(parents=True, exist_ok=True)

    BROWSER: Browser = Browser(os.getenv("AK_BROWSER", "edge").lower())
    # ★★★ HEADLESSのデフォルトを "false" に変更 ★★★
    HEADLESS: bool = os.getenv("AK_HEADLESS", "false").lower() in {"true", "1", "yes"}
    LOG_LEVEL: LogLevel = LogLevel(os.getenv("AK_LOG_LEVEL", "INFO").upper())
    LOG_FORMAT: str = os.getenv("AK_LOG_FORMAT", "[%(asctime)s] %(levelname)-8s | %(message)s")
    STAGE: Literal["test", "staging", "prod"] = os.getenv("AK_STAGE", "test")
    SITES: Dict[str, Any] = _load_layered_site_config(CONFIG_DIR)
    DB_URL: str = getattr(Secrets, "DB_URL", _sqlite_uri(DATA_DIR / "app.db"))
    MINIMAX_API_KEY: Optional[str] = getattr(Secrets, "MINIMAX_API_KEY", None)
    GLM_API_KEY: Optional[str] = getattr(Secrets, "GLM_API_KEY", None)

def setup_logging() -> None:
    logging.basicConfig(level=Config.LOG_LEVEL.value, format=Config.LOG_FORMAT, stream=sys.stdout)
    logging.getLogger("playwright").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

def log_api_keys_visibility() -> None:
    logger = logging.getLogger(__name__)
    def mask(v: Optional[str]) -> str: return f"{v[:4]}***" if v else "Not Set"
    logger.info("--- API Key Diagnostics ---")
    logger.info(f"  MINIMAX_API_KEY: {mask(Config.MINIMAX_API_KEY)}")
    logger.info(f"  GLM_API_KEY:     {mask(Config.GLM_API_KEY)}")
    logger.info("---------------------------")
