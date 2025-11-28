#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1.2 テスト実行スクリプト（直接実行版）
"""

import sys
import os
from pathlib import Path
import asyncio

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# 仮想環境のsite-packagesを追加（あれば）
venv_paths = [
    project_root / "venv" / "lib" / "python3.*" / "site-packages",
    project_root / ".venv" / "lib" / "python3.*" / "site-packages",
    project_root / "myenv" / "Lib" / "site-packages",
]

import glob
for venv_site_packages in venv_paths:
    matches = glob.glob(str(venv_site_packages))
    if matches:
        sys.path.insert(0, matches[0])
        break

def test_import_session_manager():
    """SessionManager の import をテスト"""
    try:
        from app.agents.browser.session_manager import SessionManager, EXTERNAL_BLOCKLIST_HOSTS
        print("✅ SessionManager import OK")
        return True
    except Exception as e:
        print(f"❌ SessionManager import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_import_browser_use_agent():
    """BrowserUseAgent の import をテスト（SessionManager 経由でアクセス）"""
    try:
        from app.agents.browser_use_agent import BrowserUseAgent
        print("✅ BrowserUseAgent import OK")
        return True
    except Exception as e:
        print(f"❌ BrowserUseAgent import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_no_duplicate_methods():
    """BrowserUseAgent に重複メソッドが残っていないことを確認"""
    try:
        import inspect
        from app.agents.browser_use_agent import BrowserUseAgent
        
        # 削除すべきメソッドが存在しないことを確認
        removed_methods = [
            "_build_context_options",
            "_setup_routes",
            "_get_session_file",
            "_apply_saved_session",
            "_setup_init_scripts",
        ]
        
        agent_methods = [name for name, _ in inspect.getmembers(BrowserUseAgent, predicate=inspect.isfunction)]
        agent_methods += [name for name, _ in inspect.getmembers(BrowserUseAgent, predicate=inspect.ismethod)]
        
        found_removed = []
        for method_name in removed_methods:
            if method_name in agent_methods:
                found_removed.append(method_name)
        
        if found_removed:
            print(f"❌ 削除すべきメソッドが残っています: {found_removed}")
            return False
        else:
            print("✅ BrowserUseAgent から重複メソッドが削除されています")
            return True
    except Exception as e:
        print(f"❌ 重複メソッドチェック失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_settings_no_duplicate_functions():
    """settings.py に重複関数が残っていないことを確認"""
    try:
        import inspect
        from app.agents.browser import settings
        
        # 削除すべき関数が存在しないことを確認
        removed_functions = [
            "setup_routes",
            "get_session_file",
            "apply_saved_session",
            "setup_init_scripts",
            "build_context_options",
        ]
        
        settings_functions = [name for name, _ in inspect.getmembers(settings, predicate=inspect.isfunction)]
        
        found_removed = []
        for func_name in removed_functions:
            if func_name in settings_functions:
                found_removed.append(func_name)
        
        if found_removed:
            print(f"❌ 削除すべき関数が残っています: {found_removed}")
            return False
        else:
            print("✅ settings.py から重複関数が削除されています")
            return True
    except Exception as e:
        print(f"❌ 重複関数チェック失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_session_manager_has_methods():
    """SessionManager に必要なメソッドが存在することを確認"""
    try:
        import inspect
        from app.agents.browser.session_manager import SessionManager
        
        # 存在すべきメソッド
        required_methods = [
            "_build_context_options",
            "_setup_routes",
            "_setup_init_scripts",
            "_apply_saved_session",
            "_get_session_file",
        ]
        
        manager_methods = [name for name, _ in inspect.getmembers(SessionManager, predicate=inspect.isfunction)]
        manager_methods += [name for name, _ in inspect.getmembers(SessionManager, predicate=inspect.ismethod)]
        
        missing_methods = []
        for method_name in required_methods:
            if method_name not in manager_methods:
                missing_methods.append(method_name)
        
        if missing_methods:
            print(f"❌ SessionManager に必要なメソッドが不足しています: {missing_methods}")
            return False
        else:
            print("✅ SessionManager に必要なメソッドがすべて存在します")
            return True
    except Exception as e:
        print(f"❌ SessionManager メソッドチェック失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_session_manager_basic():
    """SessionManager の基本的な動作をテスト（モック使用）"""
    try:
        from app.agents.browser.session_manager import SessionManager
        from app.core.run_context import RunContext
        from pathlib import Path
        import tempfile
        
        # 一時ディレクトリを作成
        with tempfile.TemporaryDirectory() as tmpdir:
            run_ctx = RunContext(base_path=str(Path(tmpdir) / "runs"))
            
            # SessionManager インスタンスを作成（実際の起動はしない）
            mgr = SessionManager(
                site="MONCLER_OFFICIAL",
                site_config={"discovery_settings": {}},
                run_context=run_ctx,
                settings={"timeout_sec": 1},
                target_url="https://example.com",
                timeout_ms=1000,
            )
            
            # インスタンスが作成できることを確認
            assert mgr is not None
            assert mgr.site == "MONCLER_OFFICIAL"
            print("✅ SessionManager インスタンス作成 OK")
            return True
    except Exception as e:
        print(f"❌ SessionManager 基本動作テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン処理"""
    print("=" * 80)
    print("Stage 1.2 テスト実行")
    print("=" * 80)
    print()
    
    results = []
    
    print("1. SessionManager import テスト")
    results.append(test_import_session_manager())
    print()
    
    print("2. BrowserUseAgent import テスト")
    results.append(test_import_browser_use_agent())
    print()
    
    print("3. BrowserUseAgent 重複メソッドチェック")
    results.append(test_no_duplicate_methods())
    print()
    
    print("4. settings.py 重複関数チェック")
    results.append(test_settings_no_duplicate_functions())
    print()
    
    print("5. SessionManager メソッド存在チェック")
    results.append(test_session_manager_has_methods())
    print()
    
    print("6. SessionManager 基本動作テスト")
    try:
        result = asyncio.run(test_session_manager_basic())
        results.append(result)
    except Exception as e:
        print(f"❌ 非同期テスト実行失敗: {e}")
        results.append(False)
    print()
    
    print("=" * 80)
    if all(results):
        print("✅ すべてのテストが成功しました")
        print("Stage 1.2 の実装は正常に動作しています。")
        return 0
    else:
        print("❌ 一部のテストが失敗しました")
        return 1

if __name__ == "__main__":
    sys.exit(main())

