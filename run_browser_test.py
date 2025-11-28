#!/usr/bin/env python3
"""
実ブラウザテスト実行スクリプト
WSL環境での出力取得を改善
"""
import asyncio
import sys
import os
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def main():
    """BrowserUseAgent の実ブラウザテストを実行"""
    try:
        from tools.run_browser_use import main as run_main
        print("=" * 80)
        print("実ブラウザテスト開始: MONCLER_OFFICIAL")
        print("=" * 80)
        
        # コマンドライン引数を設定
        sys.argv = [
            "run_browser_use.py",
            "--site", "MONCLER_OFFICIAL",
            "--url", "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB",
            "--query", "down jacket",
            "--headful",
            "--timeout", "120"
        ]
        
        await run_main()
        print("=" * 80)
        print("テスト完了")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n[中断] ユーザーによる中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n[エラー] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[中断] ユーザーによる中断")
        sys.exit(130)

