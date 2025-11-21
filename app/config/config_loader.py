# ==============================================================================
# File: app/config/config_loader.py
# Date (JST): 2025-10-26 19:05
# Version: 1.0 (Compat shim for legacy import path)
#
# 目的:
# - 既存コードが "app.config.config_loader" を import する前提を満たすための互換レイヤ。
# - 本来の設定ローダは app/config/loader.py にあるため、必要最小限のAPIを委譲提供する。
# - さらに、Secrets/.env/環境変数から任意キーを取得できる Config.get() を提供。
# ==============================================================================

from __future__ import annotations
from pathlib import Path
import os

# 正式なローダ機能（sites/base.json 等のマージ）を委譲
try:
    from .loader import load_full_config, get_site_config, load_and_merge_configs  # re-export
except Exception:
    # loader.py が何らかの理由で読めない場合でも import エラーにしない
    def load_full_config(*args, **kwargs):
        return {}
    def get_site_config(*args, **kwargs):
        return None
    def load_and_merge_configs(*args, **kwargs):
        return {}

# Secrets（generate_secrets.py が出力）を最優先に読む
try:
    from .secrets import Secrets  # type: ignore
except Exception:
    Secrets = None  # type: ignore


class Config:
    """
    互換API: Config.get("KEY", default)
      優先順位: secrets.py > .env > OS環境変数
    """
    @staticmethod
    def get(key: str, default: str | None = None) -> str | None:
        # 1) secrets.py
        if Secrets is not None and hasattr(Secrets, key):
            val = getattr(Secrets, key)
            if isinstance(val, str) and val != "":
                return val

        # 2) .env（プロジェクトルート直下）
        env_path = Path(__file__).resolve().parents[2] / ".env"
        if env_path.exists():
            try:
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    if k.strip() == key:
                        return v.strip()
            except Exception:
                pass

        # 3) OS 環境変数
        return os.environ.get(key, default)
