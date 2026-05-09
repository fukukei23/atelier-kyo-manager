#!/usr/bin/env python3
"""
test_session_manager.py を直接実行するスクリプト
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# pytest をインポートして実行
try:
    import pytest

    # テストファイルのパス
    test_file = project_root / "tests" / "test_session_manager.py"

    if not test_file.exists():
        print(f"❌ テストファイルが見つかりません: {test_file}")
        sys.exit(1)

    print("=" * 80)
    print("Stage 1.2 テスト実行: test_session_manager.py")
    print("=" * 80)
    print(f"\nテストファイル: {test_file}\n")

    # pytest を実行
    exit_code = pytest.main(
        [
            str(test_file),
            "-v",
            "--tb=short",
            "-s",  # 出力を表示
        ]
    )

    print()
    print("=" * 80)
    if exit_code == 0:
        print("✅ すべてのテストが成功しました")
    else:
        print(f"❌ テストが失敗しました (終了コード: {exit_code})")
    print("=" * 80)

    sys.exit(exit_code)

except ImportError:
    print("❌ pytest がインストールされていません。")
    print("以下のコマンドでインストールしてください:")
    print("  pip install pytest")
    sys.exit(1)
except Exception as e:
    print(f"❌ エラーが発生しました: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
