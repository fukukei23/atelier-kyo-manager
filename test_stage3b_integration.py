#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3B 統合テスト: TelemetryService と NavigationDriver の統合確認
"""

import sys
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from app.agents.browser.telemetry import (
    TelemetryService,
    RunPhase,
    FailureContext,
)
from app.agents.browser.navigation_driver import (
    NavigationDriver,
    NavigationContext,
    NavigationOutcome,
)
from app.core.run_context import RunContext


async def test_telemetry_service_basic():
    """TelemetryService の基本動作確認"""
    print("\n[1] TelemetryService の基本動作確認...")
    
    run_context = RunContext(base_path="instance/test_runs")
    telemetry = TelemetryService(run_context=run_context)
    
    # record_success の確認
    await telemetry.record_success(
        phase=RunPhase.PLP_DISCOVERY,
        url="https://example.com",
    )
    
    success_files = list(run_context.run_path.glob("success_*.json"))
    assert len(success_files) > 0, "Success file should be created"
    print(f"✓ record_success 成功 (ファイル: {success_files[0].name})")
    
    # record_failure の確認
    failure = FailureContext(
        site_code="TEST",
        url="https://example.com",
        phase=RunPhase.PLP_DISCOVERY,
        exception=ValueError("Test error"),
    )
    
    mock_page = MagicMock()
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.content = AsyncMock(return_value="<html></html>")
    mock_page.locator = MagicMock()
    
    mock_locator = MagicMock()
    mock_locator.count = AsyncMock(return_value=0)
    mock_page.locator.return_value = mock_locator
    
    run_context.take_screenshot = AsyncMock(return_value="failure.png")
    
    await telemetry.record_failure(failure, page=mock_page)
    
    snapshot_file = run_context.run_path / "fail_snapshot.md"
    assert snapshot_file.exists(), "Failure snapshot should be created"
    print(f"✓ record_failure 成功 (ファイル: {snapshot_file.name})")
    
    return True


async def test_navigation_driver_with_telemetry():
    """NavigationDriver と TelemetryService の統合確認"""
    print("\n[2] NavigationDriver と TelemetryService の統合確認...")
    
    run_context = RunContext(base_path="instance/test_runs")
    telemetry = TelemetryService(run_context=run_context)
    
    # モック Page オブジェクト
    mock_page = MagicMock()
    mock_page.url = "https://www.moncler.com/en-int/men/"
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.content = AsyncMock(return_value="<html><body>Test</body></html>")
    mock_page.locator = MagicMock()
    
    mock_locator = MagicMock()
    mock_locator.count = AsyncMock(return_value=5)
    mock_page.locator.return_value = mock_locator
    
    # モック関数
    async def mock_ensure_plp_materialized(page, site_config, settings, start_t, budget_ms):
        return True
    
    # NavigationDriver の作成
    nav_driver = NavigationDriver(
        page=mock_page,
        ensure_plp_materialized=mock_ensure_plp_materialized,
        trap_checker=None,
        recovery_fn=None,
        telemetry=telemetry,  # TelemetryService を渡す
        strategy=None,
    )
    
    # NavigationContext の作成
    nav_ctx = NavigationContext(
        site="TEST",
        query="GUCCI",
        site_config={"selectors": {"pdp": {}}},
        settings={"timeout_sec": 30},
        run_context=run_context,
        start_t=0.0,
        budget_ms=30000,
        entry_url="https://www.moncler.com/en-int/men/",
    )
    
    # run_plp_flow の実行
    outcome = await nav_driver.run_plp_flow(nav_ctx)
    
    assert isinstance(outcome, NavigationOutcome), "Should return NavigationOutcome"
    assert outcome.entry_url == "https://www.moncler.com/en-int/men/", "Entry URL should match"
    print(f"✓ NavigationDriver.run_plp_flow 成功")
    print(f"  - plp_materialized: {outcome.plp_materialized}")
    print(f"  - trap_detected: {outcome.trap_detected}")
    print(f"  - recovered: {outcome.recovered}")
    
    return True


async def test_telemetry_compatibility_methods():
    """TelemetryService の互換性メソッドの確認"""
    print("\n[3] TelemetryService の互換性メソッドの確認...")
    
    run_context = RunContext(base_path="instance/test_runs")
    telemetry = TelemetryService(run_context=run_context)
    
    # save_dom の確認
    mock_page = MagicMock()
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.content = AsyncMock(return_value="<html></html>")
    
    await telemetry.save_dom(mock_page, "test_dom")
    
    dom_file = run_context.run_path / "test_dom.html"
    assert dom_file.exists(), "DOM file should be created"
    print(f"✓ save_dom 成功 (ファイル: {dom_file.name})")
    
    # count_selectors の確認
    mock_page2 = MagicMock()
    mock_page2.is_closed = MagicMock(return_value=False)
    mock_page2.locator = MagicMock()
    
    mock_locator = MagicMock()
    mock_locator.count = AsyncMock(return_value=3)
    mock_page2.locator.return_value = mock_locator
    
    await telemetry.count_selectors(mock_page2, ["div.test"], name="test_counts")
    
    counts_file = run_context.run_path / "test_counts.json"
    assert counts_file.exists(), "Counts file should be created"
    print(f"✓ count_selectors 成功 (ファイル: {counts_file.name})")
    
    # save_raw_hrefs の確認
    hrefs = ["https://example.com/product1", "https://example.com/product2"]
    await telemetry.save_raw_hrefs(hrefs, name="test_hrefs")
    
    hrefs_file = run_context.run_path / "test_hrefs.json"
    assert hrefs_file.exists(), "Hrefs file should be created"
    print(f"✓ save_raw_hrefs 成功 (ファイル: {hrefs_file.name})")
    
    return True


async def main():
    """メイン実行関数"""
    print("=" * 60)
    print("Stage 3B 統合テスト")
    print("=" * 60)
    
    try:
        # テスト1: TelemetryService の基本動作
        await test_telemetry_service_basic()
        
        # テスト2: NavigationDriver と TelemetryService の統合
        await test_navigation_driver_with_telemetry()
        
        # テスト3: 互換性メソッド
        await test_telemetry_compatibility_methods()
        
        print("\n" + "=" * 60)
        print("✓ すべてのテストが成功しました！")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ 致命的なエラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

