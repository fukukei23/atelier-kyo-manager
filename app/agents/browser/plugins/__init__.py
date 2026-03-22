# -*- coding: utf-8 -*-
"""
Plugin自動登録システム
======================================================================
BrowserUseAgentが動的にサイト別戦略Pluginを読み込めるようにする。
======================================================================
"""
from __future__ import annotations

import logging
import importlib
import pkgutil
from typing import Dict, Optional, Type, List

logger = logging.getLogger(__name__)


def _discover_plugins() -> List[Type]:
    """
    app/agents/plugins/ 配下のすべてのPluginクラスを自動検出する。

    検出条件:
    - StrategyPlugin を継承している
    - クラス名が *_Strategy で終わる
    """
    discovered: List[Type] = []
    base_module = "app.agents.plugins"

    try:
        # pluginsパッケージを取得
        pkg = importlib.import_module(base_module)
        if not hasattr(pkg, '__path__'):
            return discovered

        # 各モジュールを走査
        for _, module_name, is_pkg in pkgutil.iter_modules(pkg.__path__, f"{base_module}."):
            if is_pkg:
                continue  # パッケージはスキップ

            try:
                module = importlib.import_module(module_name)
                # モジュール内のクラスを走査
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if not isinstance(attr, type):
                        continue
                    # StrategyPluginを継承しているかチェック
                    # site属性がある类是Plugin
                    if (attr_name.endswith("Strategy") and
                        not attr_name.startswith("_") and
                        hasattr(attr, 'site') and
                        isinstance(getattr(attr, 'site', None), str)):
                        discovered.append(attr)
                        logger.info(f"[PluginRegistry] Discovered: {attr_name} (site: {attr.site})")
            except Exception as e:
                logger.warning(f"[PluginRegistry] Failed to import {module_name}: {e}")

    except Exception as e:
        logger.error(f"[PluginRegistry] Plugin discovery failed: {e}")

    return discovered


# 遅延読み込みで循環参照を避ける
_PLUGINS: Optional[Dict[str, object]] = None


def get_plugin_registry() -> Dict[str, object]:
    """Pluginレジストリを返す（遅延読み込み）"""
    global _PLUGINS
    if _PLUGINS is not None:
        return _PLUGINS

    _PLUGINS = {}

    try:
        from app.agents.plugins.base import StrategyPlugin

        # 手動登録（後方互換性）
        try:
            from app.agents.plugins.moncler_plp_v1 import MonclerPLPStrategy
            _PLUGINS["MONCLER_OFFICIAL"] = MonclerPLPStrategy()
            logger.info("[PluginRegistry] Registered MONCLER_OFFICIAL")
        except ImportError:
            pass

        # 自動検出
        discovered = _discover_plugins()
        for plugin_cls in discovered:
            try:
                instance = plugin_cls()
                site_name = getattr(instance, 'site', None)
                if site_name:
                    _PLUGINS[site_name] = instance
                    logger.info(f"[PluginRegistry] Auto-registered {site_name}: {plugin_cls.__name__}")
            except Exception as e:
                logger.warning(f"[PluginRegistry] Failed to instantiate {plugin_cls.__name__}: {e}")

        logger.info(f"[PluginRegistry] Total plugins registered: {len(_PLUGINS)}")

    except ImportError as e:
        logger.warning(f"[PluginRegistry] Could not import StrategyPlugin: {e}")

    return _PLUGINS


def get_plugin_for_site(site_name: str) -> Optional[object]:
    """指定されたサイト名に対応するPluginを返す"""
    registry = get_plugin_registry()
    return registry.get(site_name.upper()) or registry.get(site_name)


# 旧形式との後方互換性
PLUGIN_REGISTRY = get_plugin_registry()
