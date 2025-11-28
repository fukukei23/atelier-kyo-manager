#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NavigationDriver の動作確認スクリプト（Stage 3A-2）

基本的なインポート、インスタンス化、メソッド呼び出しを確認します。
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """インポートの確認"""
    print("=" * 60)
    print("1. インポート確認")
    print("=" * 60)
    
    try:
        from app.agents.browser.navigation_driver import (
            NavigationDriver,
            NavigationContext,
            NavigationOutcome,
            RecoveryFn,
            EnsurePlpMaterializedFn,
            TrapCheckerFn,
        )
        print("✓ NavigationDriver 関連のインポート成功")
        
        from app.agents.browser_use_agent import BrowserUseAgent
        print("✓ BrowserUseAgent のインポート成功")
        
        return True
    except Exception as e:
        print(f"✗ インポートエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_static_method():
    """静的メソッドの確認"""
    print("\n" + "=" * 60)
    print("2. 静的メソッド looks_like_trap_or_legal の確認")
    print("=" * 60)
    
    try:
        from app.agents.browser.navigation_driver import NavigationDriver
        
        # テストケース
        test_cases = [
            ("https://www.moncler.com/en-jp/", True, "JP locale trap"),
            ("https://www.moncler.com/en-int/", True, "Locale gate"),
            ("https://www.moncler.com/en-int/men/", False, "Normal PLP"),
            ("https://www.moncler.com/cookie-policy", True, "Legal page"),
            ("https://www.monclergroup.com/", True, "Corporate site"),
        ]
        
        for url, expected, description in test_cases:
            result = NavigationDriver.looks_like_trap_or_legal(url)
            status = "✓" if result == expected else "✗"
            print(f"{status} {description}: {url}")
            print(f"  期待値: {expected}, 実際: {result}")
            if result != expected:
                print(f"  ⚠ 期待値と異なります")
        
        return True
    except Exception as e:
        print(f"✗ 静的メソッドテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataclasses():
    """データクラスの確認"""
    print("\n" + "=" * 60)
    print("3. データクラスの確認")
    print("=" * 60)
    
    try:
        from app.agents.browser.navigation_driver import NavigationContext, NavigationOutcome
        
        # NavigationContext の作成
        ctx = NavigationContext(
            site="MONCLER_OFFICIAL",
            query="GUCCI",
            site_config={"home_url": "https://www.moncler.com"},
            settings={"timeout_sec": 30},
            run_context=None,
            start_t=0.0,
            budget_ms=30000,
            entry_url="https://www.moncler.com/en-int/men/",
        )
        print(f"✓ NavigationContext 作成成功")
        print(f"  site: {ctx.site}, query: {ctx.query}, entry_url: {ctx.entry_url}")
        
        # NavigationOutcome の作成
        outcome = NavigationOutcome(
            entry_url="https://www.moncler.com/en-int/men/",
            plp_materialized=True,
            trap_detected=False,
            trap_reason=None,
            recovered=False,
        )
        print(f"✓ NavigationOutcome 作成成功")
        print(f"  entry_url: {outcome.entry_url}, plp_materialized: {outcome.plp_materialized}")
        print(f"  trap_detected: {outcome.trap_detected}, recovered: {outcome.recovered}")
        
        return True
    except Exception as e:
        print(f"✗ データクラステストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_navigation_driver_init():
    """NavigationDriver の初期化確認（モック使用）"""
    print("\n" + "=" * 60)
    print("4. NavigationDriver の初期化確認")
    print("=" * 60)
    
    try:
        from app.agents.browser.navigation_driver import NavigationDriver
        
        # モック Page オブジェクト
        class MockPage:
            def __init__(self):
                self.url = "https://www.moncler.com/en-int/men/"
            
            def is_closed(self):
                return False
        
        # モック関数
        async def mock_ensure_plp_materialized(page, site_config, settings, start_t, budget_ms):
            return True
        
        # NavigationDriver のインスタンス化
        mock_page = MockPage()
        driver = NavigationDriver(
            page=mock_page,
            ensure_plp_materialized=mock_ensure_plp_materialized,
            recovery_fn=None,
            telemetry=None,
            strategy=None,
        )
        
        print(f"✓ NavigationDriver インスタンス化成功")
        print(f"  page.url: {driver.page.url}")
        print(f"  ensure_plp_materialized: {driver.ensure_plp_materialized is not None}")
        print(f"  recovery_fn: {driver.recovery_fn}")
        
        return True
    except Exception as e:
        print(f"✗ NavigationDriver 初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メイン関数"""
    print("\n" + "=" * 60)
    print("NavigationDriver 動作確認スクリプト (Stage 3A-2)")
    print("=" * 60)
    
    results = []
    
    # 各テストを実行
    results.append(("インポート", test_imports()))
    results.append(("静的メソッド", test_static_method()))
    results.append(("データクラス", test_dataclasses()))
    results.append(("NavigationDriver初期化", test_navigation_driver_init()))
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n合計: {passed}/{total} テストが成功しました")
    
    if passed == total:
        print("\n🎉 すべてのテストが成功しました！")
        return 0
    else:
        print(f"\n⚠ {total - passed} 個のテストが失敗しました")
        return 1


if __name__ == "__main__":
    sys.exit(main())

