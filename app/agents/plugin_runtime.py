# -*- coding: utf-8 -*-
# File: app/agents/plugin_runtime.py
# Version: 0.1.0
# Purpose: plugins 配下から StrategyPlugin を動的ロード

import importlib
import pkgutil
import logging
from typing import List, Optional, Type
from .plugins.base import StrategyPlugin

logger = logging.getLogger(__name__)

def _iter_plugin_modules():
    pkg_name = "app.agents.plugins"
    pkg = importlib.import_module(pkg_name)
    for m in pkgutil.iter_modules(pkg.__path__, prefix=pkg_name + "."):
        # base.py 自体はスキップ
        if m.name.endswith(".base"):
            continue
        yield m.name

def load_plugins_for_site(site: str) -> List[StrategyPlugin]:
    """site名に合致する StrategyPlugin を全探索ロード。"""
    instances: List[StrategyPlugin] = []
    for mod_name in _iter_plugin_modules():
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            logger.warning(f"[Plugin] import failed: {mod_name}: {e}")
            continue

        # モジュール内の StrategyPlugin の具象クラスを探す
        for attr in dir(mod):
            obj = getattr(mod, attr)
            try:
                # クラスかつ StrategyPlugin のサブクラスで site が一致
                if isinstance(obj, type) and issubclass(obj, StrategyPlugin):
                    if getattr(obj, "site", "") == site:
                        instances.append(obj())  # インスタンス化
            except Exception:
                continue
    if instances:
        logger.info(f"[Plugin] Loaded {len(instances)} plugin(s) for site={site}")
    return instances

def first_plugin_for_site(site: str) -> Optional[StrategyPlugin]:
    """最初に見つかったプラグイン1つ（単一戦略運用時）。"""
    items = load_plugins_for_site(site)
    return items[0] if items else None
