# ==============================================================================
# File: app/config/config_loader.py
# Date (JST): 2025-12-01
# Version: 2.0 (NexusCore-style separation)
#
# 目的:
# - 既存コードが "app.config.config_loader" を import する前提を満たすための互換レイヤ
# - サイト設定ローダ（loader.py）のAPIを委譲提供
# - Secrets/.env/環境変数から任意キーを取得できる Config.get() を提供（後方互換性）
# ==============================================================================

from __future__ import annotations

import os
from pathlib import Path

# 正式なローダ機能（sites/base.json 等のマージ）を委譲
try:
    from .loader import get_site_config, load_and_merge_configs, load_full_config  # re-export
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

    注意: 機密情報は Secrets クラスから直接取得することを推奨
    """

    @staticmethod
    def get(key: str, default: str | None = None) -> str | None:
        """
        設定値を取得（後方互換性のため維持）

        推奨: 機密情報は `from app.config.secrets import Secrets` から直接取得
        """
        # 1) secrets.py（最優先）
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
