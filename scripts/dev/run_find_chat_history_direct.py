#!/usr/bin/env python3
"""
Cursorチャット履歴検索スクリプトを直接実行（インポート方式）
"""

import sys
from pathlib import Path

# スクリプトのディレクトリをパスに追加
tools_dir = Path(__file__).parent / "tools"
sys.path.insert(0, str(tools_dir))

# スクリプトを直接インポートして実行
try:
    # find_cursor_chat_historyモジュールをインポート
    import importlib.util

    script_path = tools_dir / "find_cursor_chat_history.py"

    spec = importlib.util.spec_from_file_location("find_cursor_chat_history", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # main関数を実行
    if hasattr(module, "main"):
        module.main()
    else:
        print("❌ main関数が見つかりません")
        sys.exit(1)

except Exception as e:
    print(f"❌ 実行エラー: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
