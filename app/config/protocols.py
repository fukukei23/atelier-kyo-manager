"""
app/config/protocols.py - 設定・依存の抽象化プロトコル

責務分割・結合度低減のため、設定取得をプロトコルで抽象化する。
テスト時や将来の拡張で設定ソースを差し替え可能にする。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable  # noqa: F401


@runtime_checkable
class SiteConfigProvider(Protocol):
    """サイト設定を提供するプロトコル。loader の直接呼び出しを差し替え可能にする。"""

    def get_site_config(self, site_name: str) -> dict[str, Any] | None:
        """指定サイトの設定を返す。存在しなければ None。"""
        ...

    def get_full_config(self) -> dict[str, Any]:
        """全サイトの設定辞書を返す。"""
        ...


class DefaultSiteConfigProvider:
    """loader.load_full_config / get_site_config をラップするデフォルト実装。"""

    def __init__(self) -> None:
        pass

    def get_site_config(self, site_name: str) -> dict[str, Any] | None:
        from app.config.loader import get_site_config as loader_get_site_config

        return loader_get_site_config(site_name)

    def get_full_config(self) -> dict[str, Any]:
        from app.config.loader import load_full_config

        return load_full_config()
