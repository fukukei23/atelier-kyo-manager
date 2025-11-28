#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WSL環境でコマンド出力を確実に取得するためのラッパースクリプト
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command_with_output(cmd, timeout=300):
    """コマンドを実行して出力を取得"""
    project_root = Path(__file__).parent
    output_file = project_root / ".test_output.txt"
    error_file = project_root / ".test_error.txt"
    
    # 環境変数を設定
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    
    try:
        # コマンドを実行してファイルにリダイレクト
        with open(output_file, 'w', encoding='utf-8') as out_f, \
             open(error_file, 'w', encoding='utf-8') as err_f:
            
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=out_f,
                stderr=err_f,
                env=env,
                cwd=str(project_root),
                text=True,
                bufsize=0,  # バッファリングなし
            )
            
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                return None, f"タイムアウト ({timeout}秒)"
        
        # ファイルから読み取る
        output = ""
        error = ""
        
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                output = f.read()
        
        if error_file.exists():
            with open(error_file, 'r', encoding='utf-8') as f:
                error = f.read()
        
        return output, error, process.returncode
        
    except Exception as e:
        return None, str(e), -1
    finally:
        # 一時ファイルをクリーンアップ
        if output_file.exists():
            try:
                output_file.unlink()
            except:
                pass
        if error_file.exists():
            try:
                error_file.unlink()
            except:
                pass


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python run_test_with_output.py <コマンド>")
        sys.exit(1)
    
    cmd = " ".join(sys.argv[1:])
    print(f"実行コマンド: {cmd}")
    print("=" * 70)
    
    output, error, returncode = run_command_with_output(cmd)
    
    if output:
        print("=== 標準出力 ===")
        print(output)
    
    if error:
        print("=== 標準エラー出力 ===")
        print(error)
    
    print("=" * 70)
    print(f"終了コード: {returncode}")
    
    sys.exit(returncode if returncode is not None else 1)

