# ==============================================================================
# ファイル名 (File Name): loader.py
# レジストリ (Registry): app/config/loader.py
# 更新日時 (Date & Time JST): 2025年10月10日 16:57:00
# バージョン (Version): 3.1.0J (Logic Restoration & Pinned Integration)
#
# --- v3.1.0Jでの主な変更点 ---
# - [重大なバグ修正] あなたの完璧な指摘に基づき、v3.0.0Jで欠落していた
#   `default_discovery_settings` の適用ロジックを完全に復元しました。
# - [Pinnedレイヤーの再統合] 復元された正しいアーキテクチャの上に、
#   `sites_pinned` レイヤーを最終的な優先権を持つ形で再統合しました。
#
# --- 既存機能からの維持 ---
# - v2.1.0Jの全てのコアロジックと後方互換性エイリアスは完全に維持されています。
# ==============================================================================
# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import logging
import copy
from typing import Dict, Any, List, Iterable
from pathlib import Path

# --- モジュールとして公開する関数を明示 ---
__all__ = ["load_full_config", "get_site_config", "load_and_merge_configs"]

logger = logging.getLogger(__name__)
APP_ROOT = Path(__file__).resolve().parents[2]
SITES_DIR = APP_ROOT / "app" / "config" / "sites"
PINNED_DIR = APP_ROOT / "app" / "config" / "sites_pinned"

# --- 内部キャッシュ ---
_cached_config: Dict[str, Any] | None = None

def deep_merge(source: Dict, destination: Dict) -> Dict:
    """深い辞書マージを行うヘルパー関数。"""
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            deep_merge(value, node)
        else:
            destination[key] = value
    return destination

def load_full_config(force_reload: bool = False) -> Dict[str, Any]:
    """
    設定を3階層でマージ: base -> overrides.local -> pinned
    """
    global _cached_config
    if _cached_config is not None and not force_reload:
        logger.debug("Returning cached configuration.")
        return _cached_config

    final_config: Dict[str, Any] = {}
    default_discovery_settings: Dict[str, Any] = {}

    try:
        # 1. base.jsonの読み込みとデフォルト設定の適用
        base_path = SITES_DIR / "base.json"
        if base_path.exists():
            logger.info(f"Loading base configuration from: {base_path}")
            with base_path.open("r", encoding="utf-8") as f:
                base_data = json.load(f)

            sites_list: List[Dict[str, Any]] = base_data.get("sites", [])
            default_discovery_settings = base_data.get("default_discovery_settings", {})
            base_sites_dict = {site["name"]: site for site in sites_list}

            for site_name, site_config in base_sites_dict.items():
                merged_site_config = copy.deepcopy(site_config)
                discovery_settings = merged_site_config.get("discovery_settings", {})
                merged_discovery_settings = deep_merge(copy.deepcopy(default_discovery_settings), discovery_settings)
                merged_site_config["discovery_settings"] = merged_discovery_settings
                final_config[site_name] = merged_site_config

        # 2. overrides.local.json の読み込みとマージ
        override_path = SITES_DIR / "overrides.local.json"
        if override_path.exists():
            logger.info(f"Loading and merging overrides from: {override_path}")
            with override_path.open("r", encoding="utf-8") as f:
                overrides = json.load(f)

            for site_name, site_override in overrides.items():
                target_node = final_config.get(site_name)
                if target_node:
                    deep_merge(site_override, target_node)
                else:
                    new_site_config = {}
                    if default_discovery_settings:
                         new_site_config["discovery_settings"] = copy.deepcopy(default_discovery_settings)
                    deep_merge(site_override, new_site_config)
                    final_config[site_name] = new_site_config
            logger.info("Successfully merged local override configurations.")

        # 3. pinned/*.json の読み込みと最終マージ
        if PINNED_DIR.exists():
            pinned_files: Iterable[Path] = sorted(PINNED_DIR.glob("*.json"))
            if pinned_files:
                logger.info("Loading pinned overrides (final layer): %s", ", ".join(p.name for p in pinned_files))
                for jf in pinned_files:
                    if jf.exists():
                        with jf.open("r", encoding="utf-8") as f:
                            pinned = json.load(f)
                        for site_name, site_pinned in pinned.items():
                            target_node = final_config.get(site_name)
                            if target_node:
                                deep_merge(site_pinned, target_node)
                            else:
                                new_site_config = {}
                                if default_discovery_settings:
                                    new_site_config["discovery_settings"] = copy.deepcopy(default_discovery_settings)
                                deep_merge(site_pinned, new_site_config)
                                final_config[site_name] = new_site_config
            else:
                logger.info("Pinned dir is empty: %s", PINNED_DIR)
        else:
            logger.info("Pinned dir not found (skipped): %s", PINNED_DIR)

    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"FATAL: Failed to load or parse configuration files: {e}", exc_info=True)
        raise

    _cached_config = final_config
    return final_config

def get_site_config(site_name: str) -> Dict[str, Any] | None:
    """
    ロード済みの全設定から、指定されたサイトの設定を安全に取得する。
    """
    full_config = load_full_config()
    return full_config.get(site_name)

# --- ★★★ 後方互換性のためのエイリアス ★★★ ---
def load_and_merge_configs(force_reload: bool = False) -> Dict[str, Any]:
    """
    旧来のOrchestrator等との互換性のために維持されるエイリアス。
    新しい公式インターフェース `load_full_config` を呼び出します。
    """
    logger.warning(
        "The function 'load_and_merge_configs' is deprecated and will be removed in a future version. "
        "Please use 'load_full_config' instead."
    )
    return load_full_config(force_reload)

