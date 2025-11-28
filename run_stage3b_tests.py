#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3B テスト実行スクリプト
"""

import sys
import subprocess
from pathlib import Path

# プロジェクトルートに移動
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """メイン実行関数"""
    print("=" * 60)
    print("Stage 3B テスト実行")
    print("=" * 60)
    
    # pytest を実行
    cmd = [
        sys.executable,
        "-m", "pytest",
        "tests/test_telemetry_service_stage3b.py",
        "-v",
        "--tb=short",
    ]
    
    print(f"\n実行コマンド: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=False,
            text=True,
        )
        return result.returncode
    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

