#!/usr/bin/env python3
"""
tests/ 配下のすべてのテストを実行し、結果をファイルに出力するスクリプト
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def main():
    project_root = Path(__file__).parent
    tests_dir = project_root / "tests"
    
    # 仮想環境の有効化（WSL環境用）
    venv_activate = None
    for venv_path in [project_root / "venv", project_root / "myenv"]:
        if venv_path.exists():
            venv_activate = venv_path / "bin" / "activate"
            break
    
    print("=" * 80)
    print("テスト実行")
    print("=" * 80)
    print(f"テストディレクトリ: {tests_dir}")
    print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # ログファイル名
    log_file = project_root / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    # pytest 実行
    cmd = [
        sys.executable, "-m", "pytest",
        str(tests_dir),
        "-v",  # 詳細出力
        "--tb=short",  # トレースバックを短く
    ]
    
    print(f"実行コマンド: {' '.join(cmd)}")
    print(f"ログファイル: {log_file}")
    print()
    print("テスト実行中...")
    print()
    
    # テスト実行（標準出力とエラー出力をキャプチャ）
    try:
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("テスト実行結果\n")
            f.write("=" * 80 + "\n")
            f.write(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"コマンド: {' '.join(cmd)}\n")
            f.write("=" * 80 + "\n\n")
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                cwd=str(project_root),
            )
            
            # 出力をファイルに書き込み
            f.write(result.stdout)
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"終了コード: {result.returncode}\n")
            f.write("=" * 80 + "\n")
            
            # コンソールにも出力
            print(result.stdout)
            
            print()
            print("=" * 80)
            print(f"テスト実行完了: 終了コード {result.returncode}")
            print(f"ログファイル: {log_file}")
            print("=" * 80)
            
            return result.returncode
            
    except Exception as e:
        error_msg = f"テスト実行エラー: {e}"
        print(error_msg)
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(error_msg + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())

