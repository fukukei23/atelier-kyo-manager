# ==============================================================================
# ファイル名 (File Name): loader.py
# レジストリ (Registry): app/config/loader.py
# 更新日時 (Date & Time JST): 2025-09-20 06:44:00
# バージョン (Version): 2.1.0J (Backward Compatible)
#
# --- v2.1.0Jでの主な変更点 ---
# - [後方互換性確保] Orchestratorなど、古い関数名を呼び出す可能性がある
#   モジュールとの互換性を維持するため、`load_and_merge_configs` という
#   名前のエイリアス（別名）を追加しました。
# - [廃止予定警告] 古い関数名が呼ばれた際には、将来のバージョンで削除される
#   旨を警告ログとして出力し、段階的なコード移行を促します。
#
# --- 役割 (Role) ---
# このモジュールは、システム全体の設定管理を一元化する責務を負います。
# 3階層(base, default, overrides)をマージし、最終的な実行コンテキストを
# 生成して、すべてのエージェントに提供します。
# ==============================================================================
# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

# --- モジュールとして公開する関数を明示 ---
__all__ = ["load_full_config", "get_site_config", "load_and_merge_configs"]

logger = logging.getLogger(__name__)
APP_ROOT = Path(__file__).resolve().parents[2]

# --- 内部キャッシュ ---
_cached_config: dict[str, Any] | None = None


def deep_merge(source: dict, destination: dict) -> dict:
    """深い辞書マージを行うヘルパー関数。"""
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            deep_merge(value, node)
        else:
            destination[key] = value
    return destination


def load_full_config(force_reload: bool = False) -> dict[str, Any]:
    """
    base.json と overrides.local.json を階層的にマージして最終設定を生成する。
    システムで唯一の公式な設定読み込みインターフェース。
    """
    global _cached_config
    if _cached_config is not None and not force_reload:
        logger.debug("Returning cached configuration.")
        return _cached_config

    config_dir = APP_ROOT / "app" / "config" / "sites"
    base_path = config_dir / "base.json"
    override_path = config_dir / "overrides.local.json"

    final_config: dict[str, Any] = {}

    try:
        # 1. base.jsonの読み込み
        logger.info(f"Loading base configuration from: {base_path}")
        with base_path.open("r", encoding="utf-8") as f:
            base_data = json.load(f)

        sites_list: list[dict[str, Any]] = base_data.get("sites", [])
        default_discovery_settings: dict[str, Any] = base_data.get("default_discovery_settings", {})

        base_sites_dict = {site["name"]: site for site in sites_list}

        # 2. 各サイトにデフォルト設定を適用
        for site_name, site_config in base_sites_dict.items():
            merged_site_config = copy.deepcopy(site_config)
            discovery_settings = merged_site_config.get("discovery_settings", {})
            merged_discovery_settings = deep_merge(copy.deepcopy(default_discovery_settings), discovery_settings)
            merged_site_config["discovery_settings"] = merged_discovery_settings
            final_config[site_name] = merged_site_config

        # 3. overrides.local.json の読み込みとマージ
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
                    deep_merge(copy.deepcopy(default_discovery_settings), new_site_config)
                    deep_merge(site_override, new_site_config)
                    final_config[site_name] = new_site_config
            logger.info("Successfully merged override configurations.")

    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"FATAL: Failed to load or parse configuration files: {e}", exc_info=True)
        raise  # 設定ファイルはシステムの生命線なので、なければ起動を中止

    _cached_config = final_config
    return final_config


def get_site_config(site_name: str) -> dict[str, Any] | None:
    """
    ロード済みの全設定から、指定されたサイトの設定を安全に取得する。
    """
    full_config = load_full_config()
    return full_config.get(site_name)


# --- ★★★ 後方互換性のためのエイリアス ★★★ ---
def load_and_merge_configs(force_reload: bool = False) -> dict[str, Any]:
    """
    旧来のOrchestrator等との互換性のために維持されるエイリアス。
    新しい公式インターフェース `load_full_config` を呼び出します。
    """
    logger.warning(
        "The function 'load_and_merge_configs' is deprecated and will be removed in a future version. "
        "Please use 'load_full_config' instead."
    )
    return load_full_config(force_reload)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("--- Running Configuration Loader Test ---")

    # 統一インターフェースのテスト
    all_configs = load_full_config()
    print("\n--- Full Merged Config (via load_full_config) ---")
    print(json.dumps(all_configs, indent=2, ensure_ascii=False))

    # 後方互換性エイリアスのテスト
    all_configs_legacy = load_and_merge_configs()
    print("\n--- Full Merged Config (via DEPRECATED load_and_merge_configs) ---")
    # (ログに警告が出るはず)

    # ヘルパー関数のテスト
    ssense_config = get_site_config("SSENSE")
    print("\n--- SSENSE Specific Config (via get_site_config) ---")
    print(json.dumps(ssense_config, indent=2, ensure_ascii=False))

    logger.info("--- Test Complete ---")
