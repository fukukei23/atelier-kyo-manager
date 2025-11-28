#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NavigationDriver の動作確認（最小限）"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from app.agents.browser.navigation_driver import (
        NavigationDriver,
        NavigationContext,
        NavigationOutcome,
    )
    print("✓ インポート成功")
    
    # 静的メソッドのテスト
    result1 = NavigationDriver.looks_like_trap_or_legal("https://www.moncler.com/en-jp/")
    result2 = NavigationDriver.looks_like_trap_or_legal("https://www.moncler.com/en-int/men/")
    print(f"✓ 静的メソッド動作確認: trap={result1}, normal={result2}")
    
    # データクラスのテスト
    ctx = NavigationContext(
        site="TEST",
        query="TEST",
        site_config={},
        settings={},
        run_context=None,
        start_t=0.0,
        budget_ms=1000,
    )
    print(f"✓ NavigationContext 作成成功: site={ctx.site}")
    
    outcome = NavigationOutcome(entry_url="https://example.com")
    print(f"✓ NavigationOutcome 作成成功: entry_url={outcome.entry_url}, recovered={outcome.recovered}")
    
    print("\n✅ すべての基本チェックが成功しました！")
    sys.exit(0)
    
except Exception as e:
    print(f"❌ エラー: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

