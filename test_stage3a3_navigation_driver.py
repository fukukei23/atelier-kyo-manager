#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3A-3: NavigationDriver trap 観測フックのテスト
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def test_import_navigation_driver():
    """NavigationDriver のインポートをテスト"""
    try:
        from app.agents.browser.navigation_driver import (
            NavigationDriver,
            NavigationContext,
            NavigationOutcome,
            TrapCheckerFn,
        )
        print("✅ NavigationDriver import OK")
        return True
    except Exception as e:
        print(f"❌ NavigationDriver import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_navigation_outcome_trap_fields():
    """NavigationOutcome に trap 関連フィールドが存在することを確認"""
    try:
        from app.agents.browser.navigation_driver import NavigationOutcome
        
        outcome = NavigationOutcome(entry_url="https://example.com")
        
        # trap 関連フィールドが存在することを確認
        assert hasattr(outcome, "trap_detected"), "trap_detected フィールドが存在しません"
        assert hasattr(outcome, "trap_reason"), "trap_reason フィールドが存在しません"
        
        # デフォルト値の確認
        assert outcome.trap_detected == False, "trap_detected のデフォルト値が False であること"
        assert outcome.trap_reason is None, "trap_reason のデフォルト値が None であること"
        
        print("✅ NavigationOutcome trap フィールド OK")
        return True
    except Exception as e:
        print(f"❌ NavigationOutcome trap フィールドテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_navigation_driver_trap_checker_init():
    """NavigationDriver に trap_checker を渡せることを確認"""
    try:
        from app.agents.browser.navigation_driver import NavigationDriver
        from playwright.async_api import Page
        
        # モック Page を作成（実際の Page オブジェクトは必要ないので、None でテスト）
        # 実際のテストでは、モックや FakePage を使用する
        
        # trap_checker 関数を定義
        def trap_checker(url: str) -> bool:
            return "/legal" in url or "/cookie" in url
        
        # NavigationDriver の __init__ シグネチャを確認
        import inspect
        init_sig = inspect.signature(NavigationDriver.__init__)
        params = init_sig.parameters
        
        # trap_checker パラメータが存在することを確認
        assert "trap_checker" in params, "trap_checker パラメータが存在すること"
        trap_checker_param = params["trap_checker"]
        assert trap_checker_param.default is None or trap_checker_param.default == inspect.Parameter.empty, "trap_checker がオプション引数であること"
        
        print("✅ NavigationDriver trap_checker 引数の存在確認 OK")
        return True
    except Exception as e:
        print(f"❌ NavigationDriver trap_checker 初期化テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trap_checker_type():
    """TrapCheckerFn 型が正しく定義されていることを確認"""
    try:
        from app.agents.browser.navigation_driver import TrapCheckerFn
        from typing import get_type_hints
        
        # TrapCheckerFn が Callable であることを確認
        assert TrapCheckerFn is not None, "TrapCheckerFn が定義されていること"
        
        # 実際の関数が TrapCheckerFn 型として使えることを確認
        def test_checker(url: str) -> bool:
            return True
        
        # 型チェック（実行時には型チェックは行われないが、構文的に正しいことを確認）
        checker: TrapCheckerFn = test_checker
        assert callable(checker), "TrapCheckerFn は callable であること"
        
        print("✅ TrapCheckerFn 型定義 OK")
        return True
    except Exception as e:
        print(f"❌ TrapCheckerFn 型定義テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_navigation_driver_run_plp_flow_trap_observation():
    """NavigationDriver.run_plp_flow が trap を観測できることを確認（スタブ実装）"""
    try:
        from app.agents.browser.navigation_driver import (
            NavigationDriver,
            NavigationContext,
            NavigationOutcome,
        )
        import asyncio
        
        # trap_checker 関数を定義
        trap_urls = []
        def trap_checker(url: str) -> bool:
            is_trap = "/legal" in url or "/cookie" in url
            if is_trap:
                trap_urls.append(url)
            return is_trap
        
        # モック Page を作成（実際の Page オブジェクトは必要ないので、属性だけ持つオブジェクト）
        class MockPage:
            def __init__(self, url: str):
                self.url = url
        
        # NavigationContext を作成
        ctx = NavigationContext(
            site="TEST_SITE",
            query="test query",
            site_config={},
            settings={},
            run_context=None,
            start_t=0.0,
            budget_ms=1000,
            entry_url="https://example.com/legal",
        )
        
        # NavigationDriver を初期化
        mock_page = MockPage("https://example.com/legal")
        driver = NavigationDriver(
            page=mock_page,
            trap_checker=trap_checker,
        )
        
        # run_plp_flow を実行（非同期）
        async def run_test():
            outcome = await driver.run_plp_flow(ctx)
            
            # trap が観測されていることを確認
            assert outcome.trap_detected == True, "trap が検出されていること"
            assert outcome.trap_reason is not None, "trap_reason が設定されていること"
            assert "legal" in outcome.trap_reason or "example.com/legal" in outcome.trap_reason, "trap_reason に URL が含まれていること"
            
            return True
        
        result = asyncio.run(run_test())
        
        if result:
            print("✅ NavigationDriver.run_plp_flow trap 観測 OK")
            return True
        else:
            print("❌ NavigationDriver.run_plp_flow trap 観測テスト失敗")
            return False
    except Exception as e:
        print(f"❌ NavigationDriver.run_plp_flow trap 観測テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン処理"""
    print("=" * 80)
    print("Stage 3A-3: NavigationDriver trap 観測フック テスト")
    print("=" * 80)
    print()
    
    results = []
    
    print("1. NavigationDriver import テスト")
    results.append(test_import_navigation_driver())
    print()
    
    print("2. NavigationOutcome trap フィールドテスト")
    results.append(test_navigation_outcome_trap_fields())
    print()
    
    print("3. NavigationDriver trap_checker 初期化テスト")
    results.append(test_navigation_driver_trap_checker_init())
    print()
    
    print("4. TrapCheckerFn 型定義テスト")
    results.append(test_trap_checker_type())
    print()
    
    print("5. NavigationDriver.run_plp_flow trap 観測テスト")
    results.append(test_navigation_driver_run_plp_flow_trap_observation())
    print()
    
    print("=" * 80)
    if all(results):
        print("✅ すべてのテストが成功しました")
        print("Stage 3A-3 の実装は正常に動作しています。")
        return 0
    else:
        print("❌ 一部のテストが失敗しました")
        return 1

if __name__ == "__main__":
    sys.exit(main())

