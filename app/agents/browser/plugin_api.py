# ==============================================================================
# File: app/agents/browser/plugin_api.py
# Version: 1.0.0
# Purpose: Plugin API Facade - BrowserUseAgent から見える唯一の Plugin インターフェース
# ==============================================================================
"""
Stage 3C: Plugin API（Facade化）

BrowserUseAgent が直接 StrategyPlugin を触らないようにする。
before_navigate / after_navigate / materialize / assert_plp を Plugin API 経由で呼ぶ。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

try:
    from app.agents.plugins.base import StrategyPlugin
except Exception:
    StrategyPlugin = object  # type: ignore


@dataclass
class PluginContext:
    """Plugin に渡すコンテキスト情報（将来拡張用）"""

    site: str
    query: str
    site_config: dict[str, Any]
    settings: dict[str, Any]
    run_context: Any  # RunContext を直接 import しない（循環回避）


class PluginAPI:
    """
    BrowserUseAgent から見える唯一の Plugin Facade。

    - プラグインの取得
    - before_navigate / after_navigate / materialize / assert_plp の安全ラッパー
    - ログメッセージとフォールバックポリシーは現行実装互換
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        # PLUGIN_REGISTRY は遅延 import にする（循環 import 回避）

    def get_plugin(self, site: str) -> StrategyPlugin | None:
        """
        プラグインを取得する。

        Args:
            site: サイト名（例: "MONCLER_OFFICIAL"）

        Returns:
            StrategyPlugin インスタンス、見つからない場合は None
        """
        try:
            from app.agents.browser.plugins import PLUGIN_REGISTRY
        except Exception as e:
            self.logger.warning(f"[PluginAPI] Failed to import PLUGIN_REGISTRY: {e}")
            return None

        return PLUGIN_REGISTRY.get(str(site or "").upper())

    # --- Facade methods ---

    def before_navigate(
        self,
        plugin: StrategyPlugin | None,
        *,
        site: str,
        target_url: str,
        ctx: dict[str, Any],
    ) -> str:
        """
        before_navigate フックを安全に実行する。
        現行の try/except + ログ 挙動をそのままラップ。

        Args:
            plugin: StrategyPlugin インスタンス（None の場合は target_url をそのまま返す）
            site: サイト名（ログ用）
            target_url: 元のURL
            ctx: プラグインに渡すコンテキスト

        Returns:
            修正後のURL（失敗時は target_url を返す）
        """
        if not plugin:
            return target_url

        nav_url = target_url
        try:
            nav_url = plugin.before_navigate(target_url, ctx) or target_url
            self.logger.info(f"[Plugin:{site}] before_navigate => {nav_url}")
        except Exception as e:
            nav_url = target_url
            self.logger.warning(f"[Plugin:{site}] before_navigate failed: {e}")

        return nav_url

    async def after_navigate(
        self,
        plugin: StrategyPlugin | None,
        *,
        site: str,
        page: Any,
        ctx: dict[str, Any],
    ) -> None:
        """
        after_navigate フックを安全に実行する。

        Args:
            plugin: StrategyPlugin インスタンス（None の場合は何もしない）
            site: サイト名（ログ用）
            page: Playwright Page オブジェクト
            ctx: プラグインに渡すコンテキスト
        """
        if not plugin:
            return

        try:
            await plugin.after_navigate(page, ctx)
        except Exception as e:
            self.logger.warning(f"[Plugin:{site}] after_navigate failed: {e}")

    async def materialize_plp(
        self,
        plugin: StrategyPlugin | None,
        *,
        site: str,
        page: Any,
        ctx: dict[str, Any],
    ) -> bool | None:
        """
        materialize フックを安全に実行する。

        - 成功: True/False を返す（現行の plugin.materialize の戻り値）
        - 例外: None を返す（＝「プラグイン失敗、デフォルト PLP フロー継続」）

        Args:
            plugin: StrategyPlugin インスタンス（None の場合は None を返す）
            site: サイト名（ログ用）
            page: Playwright Page オブジェクト
            ctx: プラグインに渡すコンテキスト

        Returns:
            True: マテリアライズ成功
            False: マテリアライズ失敗（フォールバック）
            None: プラグインエラーまたはプラグインなし（フォールバック）
        """
        if not plugin:
            return None

        try:
            materialized = await plugin.materialize(page, ctx)
            if materialized:
                self.logger.info(f"[Plugin:{site}] PLP materialized successfully.")
            else:
                self.logger.warning(f"[Plugin:{site}] materialize failed (fallback to default PLP flow).")
            return materialized
        except Exception as e:
            self.logger.warning(f"[Plugin:{site}] materialize raised {e!r} (ignored, continue default PLP flow).")
            return None

    async def assert_plp(
        self,
        plugin: StrategyPlugin | None,
        *,
        site: str,
        page: Any,
        ctx: dict[str, Any],
    ) -> bool | None:
        """
        assert_plp フックを安全に実行する。

        - 成功: True/False
        - 例外: None（ログを出してデフォルト PLP フロー継続）

        Args:
            plugin: StrategyPlugin インスタンス（None の場合は None を返す）
            site: サイト名（ログ用）
            page: Playwright Page オブジェクト
            ctx: プラグインに渡すコンテキスト

        Returns:
            True: PLP と判定された
            False: PLP ではない（フォールバック）
            None: プラグインエラーまたはプラグインなし（フォールバック）
        """
        if not plugin:
            return None

        try:
            asserted = await plugin.assert_plp(page, ctx)
        except Exception as e:
            self.logger.warning(f"[Plugin:{site}] assert_plp raised {e!r} (ignored, continue default PLP flow).")
            return None
        else:
            if not asserted:
                self.logger.warning(f"[Plugin:{site}] assert_plp=False (continue with default PLP flow).")
            return asserted
