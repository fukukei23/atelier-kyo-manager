# app/config.py
# -----------------------------------------------------------------------------
# Flask 設定クラス群 + 便利ユーティリティ
# - SQLALCHEMY_DATABASE_URI を確実に設定（env なければ SQLite を自動採用）
# - CRAWLER_* 設定（ヘッドレスや待機時間など）
# - config/crawler_sites.json のロード（なければ既定の farfetch 定義を内蔵）
# - ログに設定値の要点を表示
# -----------------------------------------------------------------------------

from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict


# === ユーティリティ =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]  # プロジェクトルート(atelier-kyo-manager)
DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = BASE_DIR / "config"
CONFIG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

def _strtobool(val: Any, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    s = str(val).strip().lower()
    return s in ("1", "true", "yes", "y", "on")

def _normalize_database_url(url: str) -> str:
    # Heroku系の `postgres://` を `postgresql://` に正規化
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql://", 1)
    return url

def _load_crawler_sites_json() -> Dict[str, Any]:
    """
    config/crawler_sites.json を読み込み。
    無ければ既定の farfetch 設定を返す（最低限動くセレクタ）。
    """
    fp = CONFIG_DIR / "crawler_sites.json"
    if fp.exists():
        try:
            with fp.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"[config] failed to read {fp}: {e}")

    # 既定（最低限の Farfetch 定義）
    return {
        "farfetch": {
            # 直接検索URLを叩くテンプレ
            "search_url_template": "https://www.farfetch.com/jp/search?q={query}",
            # 検索結果ページの各商品リンク
            "search_result_link_selector": "a._5ce6a6, a._b5c581",
            # Cookie 許諾ボタン（表示されない場合もある）
            "cookie_accept_selector": "button[aria-label='Accept All Cookies'], button[aria-label='同意する']",
            # JSON-LD を含む <script> 要素（複数ある可能性）
            "structured_data_selector": "script[type='application/ld+json']"
        }
    }


# === 設定クラス =============================================================

class BaseConfig:
    """共通設定（開発/本番/テストで継承）"""
    # Flask
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-me")
    WTF_CSRF_ENABLED = True
    JSON_AS_ASCII = False

    # Database（環境変数が無い場合は SQLite を自動採用）
    _default_sqlite = f"sqlite:///{(DATA_DIR / 'app.db').as_posix()}"
    SQLALCHEMY_DATABASE_URI = _normalize_database_url(
        os.getenv("DATABASE_URL", _default_sqlite)
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

    # OpenAI / 外部APIキー（任意）
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

    # Crawler 設定
    CRAWLER_HEADLESS = _strtobool(os.getenv("CRAWLER_HEADLESS", "1"), True)
    CRAWLER_DEFAULT_WAIT_TIME = int(os.getenv("CRAWLER_DEFAULT_WAIT_TIME", "25"))
    CRAWLER_PARALLEL = _strtobool(os.getenv("CRAWLER_PARALLEL", "0"), False)
    CRAWLER_USER_DATA_DIR = os.getenv("CRAWLER_USER_DATA_DIR", "").strip() or None
    CRAWLER_DISABLE_CLIP = _strtobool(os.getenv("CRAWLER_DISABLE_CLIP", "0"), False)

    # 画像処理（背景除去などの既存処理用：必要であれば）
    IMAGE_MAX_SIZE = int(os.getenv("IMAGE_MAX_SIZE", "2048"))   # px
    IMAGE_QUALITY = int(os.getenv("IMAGE_QUALITY", "90"))       # JPEG品質

    # その他
    UPLOAD_FOLDER = (BASE_DIR / "uploads").as_posix()
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

    # ロード時に埋める（sitesは dict 想定）
    CRAWLER_TARGET_SITES: Dict[str, Any] = {}

    @classmethod
    def init_app(cls, app) -> None:
        """Flaskアプリ初期化時に追加実行する処理"""
        # ロガー初期化（既にハンドラがあれば上書きしない）
        root = logging.getLogger()
        if not root.handlers:
            logging.basicConfig(
                level=getattr(logging, cls.LOG_LEVEL, logging.INFO),
                format="%(asctime)s - %(levelname)s - %(message)s",
            )
        else:
            root.setLevel(getattr(logging, cls.LOG_LEVEL, logging.INFO))

        # サイト定義ロード
        sites = _load_crawler_sites_json()
        app.config["CRAWLER_TARGET_SITES"] = sites

        # 重要パラメータのログ出力
        logging.info(f"[config] loaded CRAWLER_TARGET_SITES from {CONFIG_DIR / 'crawler_sites.json'}")
        logging.info(
            f"[config] CRAWLER_HEADLESS={cls.CRAWLER_HEADLESS}, "
            f"PARALLEL={cls.CRAWLER_PARALLEL}, WAIT={cls.CRAWLER_DEFAULT_WAIT_TIME}s, "
            f"SITES={', '.join(sites.keys()) if isinstance(sites, dict) else 'n/a'}"
        )


class DevelopmentConfig(BaseConfig):
    DEBUG = True
    ENV = "development"

    # 開発用に SQLite をデフォルト化（環境変数があれば優先）
    _dev_sqlite = f"sqlite:///{(DATA_DIR / 'dev.db').as_posix()}"
    SQLALCHEMY_DATABASE_URI = _normalize_database_url(
        os.getenv("DATABASE_URL", _dev_sqlite)
    )


class TestingConfig(BaseConfig):
    TESTING = True
    ENV = "testing"

    # テストはメモリDBをデフォルト（環境変数があれば優先）
    SQLALCHEMY_DATABASE_URI = _normalize_database_url(
        os.getenv("DATABASE_URL", "sqlite:///:memory:")
    )
    WTF_CSRF_ENABLED = False  # テスト簡略化


class ProductionConfig(BaseConfig):
    DEBUG = False
    ENV = "production"

    # 本番は必ず DATABASE_URL を使う。なければ SQLite へフォールバック
    _prod_sqlite = f"sqlite:///{(DATA_DIR / 'prod.db').as_posix()}"
    SQLALCHEMY_DATABASE_URI = _normalize_database_url(
        os.getenv("DATABASE_URL", _prod_sqlite)
    )


# Flaskファクトリで使うマップ
config_map = {
    "default": DevelopmentConfig,
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}
