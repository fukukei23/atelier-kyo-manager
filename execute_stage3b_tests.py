#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3B テスト実行スクリプト（PowerShell問題回避版）
"""

import sys
import subprocess
from pathlib import Path

def main():
    """メイン実行関数"""
    project_root = Path(__file__).parent
    
    print("=" * 60)
    print("Stage 3B テスト実行")
    print("=" * 60)
    print(f"\nプロジェクトルート: {project_root}\n")
    
    # pytest を実行
    test_file = project_root / "tests" / "test_telemetry_service_stage3b.py"
    
    if not test_file.exists():
        print(f"❌ テストファイルが見つかりません: {test_file}")
        return 1
    
    print(f"テストファイル: {test_file}\n")
    
    # Python実行可能ファイルのパスを取得
    python_exe = sys.executable
    
    # pytest コマンドを構築
    cmd = [
        python_exe,
        "-m", "pytest",
        str(test_file),
        "-v",
        "--tb=short",
    ]
    
    print(f"実行コマンド: {' '.join(cmd)}\n")
    print("-" * 60)
    
    try:
        # subprocessで実行（shell=Falseで直接実行）
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=False,
            text=True,
        )
        
        print("-" * 60)
        print(f"\n終了コード: {result.returncode}")
        
        if result.returncode == 0:
            print("\n✅ すべてのテストが成功しました！")
        else:
            print("\n❌ 一部のテストが失敗しました。")
        
        return result.returncode
        
    except FileNotFoundError:
        print(f"❌ Python実行可能ファイルが見つかりません: {python_exe}")
        return 1
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

