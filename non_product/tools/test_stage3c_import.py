#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3C 動作確認スクリプト
PluginAPI と BrowserUseAgent の import が正常に動作するか確認
"""
from __future__ import annotations

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def test_plugin_api_import():
    """PluginAPI の import をテスト"""
    try:
        from app.agents.browser.plugin_api import PluginAPI, PluginContext
        print("✅ PluginAPI import OK")
        return True
    except Exception as e:
        print(f"❌ PluginAPI import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_browser_use_agent_import():
    """BrowserUseAgent の import をテスト"""
    try:
        from app.agents.browser_use_agent import BrowserUseAgent
        print("✅ BrowserUseAgent import OK")
        return True
    except Exception as e:
        print(f"❌ BrowserUseAgent import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_plugin_registry_import():
    """PLUGIN_REGISTRY の import をテスト（PluginAPI経由）"""
    try:
        from app.agents.browser.plugin_api import PluginAPI
        import logging
        
        logger = logging.getLogger(__name__)
        plugin_api = PluginAPI(logger)
        plugin = plugin_api.get_plugin("MONCLER_OFFICIAL")
        
        if plugin:
            print(f"✅ Plugin registry access OK (found plugin: {type(plugin).__name__})")
        else:
            print("⚠️  Plugin registry access OK (but no plugin found for MONCLER_OFFICIAL)")
        return True
    except Exception as e:
        print(f"❌ Plugin registry access failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン処理"""
    print("=" * 80)
    print("Stage 3C 動作確認")
    print("=" * 80)
    print()
    
    results = []
    
    print("1. PluginAPI import テスト")
    results.append(test_plugin_api_import())
    print()
    
    print("2. BrowserUseAgent import テスト")
    results.append(test_browser_use_agent_import())
    print()
    
    print("3. Plugin registry アクセステスト")
    results.append(test_plugin_registry_import())
    print()
    
    print("=" * 80)
    if all(results):
        print("✅ すべてのテストが成功しました")
        return 0
    else:
        print("❌ 一部のテストが失敗しました")
        return 1

if __name__ == "__main__":
    sys.exit(main())

