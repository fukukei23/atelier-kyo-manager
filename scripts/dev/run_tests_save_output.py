#!/usr/bin/env python3
"""
tests/ 配下のすべてのテストを実行し、結果をファイルに保存するスクリプト
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

# プロジェクトルート
project_root = Path(__file__).parent

# 仮想環境の Python を探す
venv_python = None
venv_paths = [
    project_root / "venv" / "bin" / "python3",
    project_root / "venv" / "bin" / "python",
    project_root / ".venv" / "bin" / "python3",
    project_root / ".venv" / "bin" / "python",
    project_root / "myenv" / "Scripts" / "python.exe",
]

for venv_python_path in venv_paths:
    if venv_python_path.exists():
        venv_python = str(venv_python_path)
        break

if not venv_python:
    venv_python = sys.executable


def main():
    # ログファイル名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = project_root / f"test_results_{timestamp}.txt"

    # pytest コマンド
    cmd = [venv_python, "-m", "pytest", "tests/", "-v", "--tb=short"]

    print("=" * 80)
    print("テスト実行")
    print("=" * 80)
    print(f"Python: {venv_python}")
    print(f"コマンド: {' '.join(cmd)}")
    print(f"ログファイル: {log_file}")
    print("=" * 80)
    print()

    try:
        # テスト実行
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("テスト実行結果\n")
            f.write("=" * 80 + "\n")
            f.write(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Python: {venv_python}\n")
            f.write(f"コマンド: {' '.join(cmd)}\n")
            f.write("=" * 80 + "\n\n")

            process = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            # 出力をファイルとコンソールに書き込み
            for line in process.stdout:
                line_text = line
                f.write(line_text)
                f.flush()
                # コンソールにも出力
                print(line_text, end="")

            process.wait()

            f.write("\n" + "=" * 80 + "\n")
            f.write(f"終了コード: {process.returncode}\n")
            f.write("=" * 80 + "\n")

            print()
            print("=" * 80)
            if process.returncode == 0:
                print("✅ すべてのテストが成功しました")
            else:
                print(f"❌ テストが失敗しました (終了コード: {process.returncode})")
            print(f"ログファイル: {log_file}")
            print("=" * 80)

            return process.returncode

    except Exception as e:
        error_msg = f"テスト実行エラー: {e}"
        print(error_msg)
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(error_msg + "\n")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
