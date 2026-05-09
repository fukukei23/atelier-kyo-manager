#!/usr/bin/env python3
"""
Cursorチャット履歴検索スクリプトを実行するラッパー
"""

import subprocess
import sys
from pathlib import Path


def main():
    script_path = Path(__file__).parent / "tools" / "find_cursor_chat_history.py"

    if not script_path.exists():
        print(f"❌ スクリプトが見つかりません: {script_path}")
        return 1

    try:
        # Pythonスクリプトを直接実行
        result = subprocess.run([sys.executable, str(script_path)], capture_output=False, text=True)
        return result.returncode
    except Exception as e:
        print(f"❌ 実行エラー: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
