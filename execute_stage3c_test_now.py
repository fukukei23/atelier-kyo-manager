#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3C 動作確認スクリプト（即座に実行版）
仮想環境を自動的に有効化してから実行します。
"""

import sys
import os
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# 仮想環境のsite-packagesを追加
def setup_venv_path():
    """仮想環境のsite-packagesをsys.pathに追加"""
    venv_dirs = [
        project_root / "venv",
        project_root / ".venv",
        project_root / "myenv",
    ]
    
    for venv_dir in venv_dirs:
        if not venv_dir.exists():
            continue
            
        # site-packages を探す
        for py_ver in ["python3.12", "python3.11", "python3.10", "python3.9", "python3"]:
            site_packages = venv_dir / "lib" / py_ver / "site-packages"
            if site_packages.exists():
                sys.path.insert(0, str(site_packages))
                print(f"📦 仮想環境を検出: {venv_dir}")
                return True
        
        # Windows用
        if sys.platform == "win32":
            site_packages = venv_dir / "Lib" / "site-packages"
            if site_packages.exists():
                sys.path.insert(0, str(site_packages))
                print(f"📦 仮想環境を検出: {venv_dir}")
                return True
    
    return False

venv_found = setup_venv_path()

def test_plugin_api_import():
    """PluginAPI の import をテスト"""
    try:
        from app.agents.browser.plugin_api import PluginAPI, PluginContext
        print("✅ PluginAPI import OK")
        return True
    except ImportError as e:
        if "flask" in str(e).lower():
            print(f"⚠️  PluginAPI import failed: {e}")
            print("   (仮想環境が有効化されていない可能性があります)")
            print("   以下のコマンドで仮想環境を有効化してください:")
            print("   source venv/bin/activate  # または source .venv/bin/activate")
            return False
        print(f"❌ PluginAPI import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
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
    except ImportError as e:
        if "flask" in str(e).lower():
            print(f"⚠️  BrowserUseAgent import failed: {e}")
            print("   (仮想環境が有効化されていない可能性があります)")
            return False
        print(f"❌ BrowserUseAgent import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
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
    except ImportError as e:
        if "flask" in str(e).lower():
            print(f"⚠️  Plugin registry access failed: {e}")
            print("   (仮想環境が有効化されていない可能性があります)")
            return False
        print(f"❌ Plugin registry access failed: {e}")
        import traceback
        traceback.print_exc()
        return False
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
    
    if not venv_found:
        print("⚠️  仮想環境が見つかりませんでした")
        print("   以下を実行してから再度テストしてください:")
        print("   source venv/bin/activate  # または source .venv/bin/activate")
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
        print()
        print("💡 ヒント:")
        print("   仮想環境を有効化してから再度実行してください:")
        print("   source venv/bin/activate  # または source .venv/bin/activate")
        return 1

if __name__ == "__main__":
    sys.exit(main())
