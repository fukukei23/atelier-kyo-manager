#!/usr/bin/env python3
"""
即座に出力を表示するテストスクリプト
WSL環境でのCursor出力取得を改善
"""
import sys
import os
import subprocess
from pathlib import Path

# バッファリングを無効にする
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def run_command_with_output(cmd, timeout=180):
    """コマンドを実行し、リアルタイムで出力を表示"""
    print(f"[実行] {' '.join(cmd)}")
    print("=" * 80)
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # 行バッファリング
        universal_newlines=True,
        cwd=Path(__file__).parent
    )
    
    try:
        # リアルタイムで出力を読み取る
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line, end='', flush=True)
        
        process.wait()
        return process.returncode
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        return 130
    except Exception as e:
        print(f"\n[エラー] {e}", file=sys.stderr, flush=True)
        process.terminate()
        return 1

if __name__ == "__main__":
    project_root = Path(__file__).parent
    
    cmd = [
        "python", "-u",  # -u: バッファリング無効
        "tools/run_browser_use.py",
        "--site", "MONCLER_OFFICIAL",
        "--url", "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB",
        "--query", "down jacket",
        "--headful",
        "--timeout", "120"
    ]
    
    exit_code = run_command_with_output(cmd, timeout=180)
    sys.exit(exit_code)

