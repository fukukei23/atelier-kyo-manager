#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3A-3 テスト実行スクリプト
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# テストスクリプトをインポートして実行
try:
    from test_stage3a3_navigation_driver import main
    sys.exit(main())
except ImportError as e:
    print(f"❌ テストスクリプトのインポートに失敗しました: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"❌ テスト実行中にエラーが発生しました: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

