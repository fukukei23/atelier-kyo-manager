#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TelemetryService のクイック動作確認スクリプト
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
from app.core.run_context import RunContext


async def main():
    """メイン実行関数"""
    print("=" * 60)
    print("TelemetryService 動作確認テスト")
    print("=" * 60)
    
    # 1. インポート確認
    print("\n[1] インポート確認...")
    assert TelemetryService is not None
    assert RunPhase is not None
    assert FailureContext is not None
    print("✓ インポート成功")
    
    # 2. RunPhase Enum確認
    print("\n[2] RunPhase Enum確認...")
    assert RunPhase.PLP_DISCOVERY.value == "plp_discovery"
    assert RunPhase.PDP_EXTRACT.value == "pdp_extract"
    print("✓ RunPhase Enum正常")
    
    # 3. FailureContext確認
    print("\n[3] FailureContext確認...")
    failure = FailureContext(
        site_code="TEST",
        url="https://example.com",
        phase=RunPhase.PLP_DISCOVERY,
    )
    assert failure.site_code == "TEST"
    assert failure.phase == RunPhase.PLP_DISCOVERY
    print("✓ FailureContext正常")
    
    # 4. TelemetryService初期化
    print("\n[4] TelemetryService初期化...")
    run_context = RunContext(base_path="instance/test_runs")
    telemetry = TelemetryService(run_context=run_context)
    assert telemetry.run_context == run_context
    assert telemetry.logger is not None
    print("✓ TelemetryService初期化成功")
    
    # 5. record_success確認
    print("\n[5] record_success動作確認...")
    await telemetry.record_success(
        phase=RunPhase.PLP_DISCOVERY,
        url="https://example.com",
    )
    success_files = list(run_context.run_path.glob("success_*.json"))
    assert len(success_files) > 0
    print(f"✓ record_success成功 (ファイル: {success_files[0].name})")
    
    # 6. record_plp_state確認（モックPage使用）
    print("\n[6] record_plp_state動作確認...")
    mock_page = MagicMock()
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.content = AsyncMock(return_value="<html><body>Test</body></html>")
    mock_page.locator = MagicMock()
    
    mock_locator = MagicMock()
    mock_locator.count = AsyncMock(return_value=5)
    mock_page.locator.return_value = mock_locator
    
    run_context.take_screenshot = AsyncMock(return_value="test.png")
    
    await telemetry.record_plp_state(
        mock_page,
        name="test_plp",
        selectors=["div.test"],
    )
    assert mock_page.content.called
    print("✓ record_plp_state成功")
    
    # 7. record_failure確認
    print("\n[7] record_failure動作確認...")
    failure = FailureContext(
        site_code="TEST",
        url="https://example.com",
        phase=RunPhase.PLP_DISCOVERY,
        exception=ValueError("Test error"),
    )
    
    mock_page2 = MagicMock()
    mock_page2.is_closed = MagicMock(return_value=False)
    mock_page2.content = AsyncMock(return_value="<html></html>")
    mock_page2.locator = MagicMock()
    
    mock_locator2 = MagicMock()
    mock_locator2.count = AsyncMock(return_value=0)
    mock_page2.locator.return_value = mock_locator2
    
    run_context.take_screenshot = AsyncMock(return_value="failure.png")
    
    await telemetry.record_failure(failure, page=mock_page2)
    
    snapshot_file = run_context.run_path / "fail_snapshot.md"
    assert snapshot_file.exists()
    print(f"✓ record_failure成功 (ファイル: {snapshot_file.name})")
    
    # 8. 内部メソッド確認
    print("\n[8] 内部メソッド確認...")
    await telemetry._save_dom(mock_page2, "test_dom")
    dom_file = run_context.run_path / "test_dom.html"
    assert dom_file.exists()
    print("✓ _save_dom成功")
    
    await telemetry._save_json({"test": "data"}, "test_json")
    json_file = run_context.run_path / "test_json.json"
    assert json_file.exists()
    print("✓ _save_json成功")
    
    print("\n" + "=" * 60)
    print("✓ すべてのテストが成功しました！")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

