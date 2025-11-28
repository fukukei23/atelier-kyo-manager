#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
テスト実行スクリプト（エイリアス）
run_pytest_direct.py へのショートカット
"""

import sys
from pathlib import Path

# run_pytest_direct.py を実行
script_path = Path(__file__).resolve().parent / "run_pytest_direct.py"

if not script_path.exists():
    print(f"❌ スクリプトが見つかりません: {script_path}")
    sys.exit(1)

# run_pytest_direct.py をインポートして実行
sys.path.insert(0, str(script_path.parent))
from run_pytest_direct import main

if __name__ == "__main__":
    sys.exit(main())

