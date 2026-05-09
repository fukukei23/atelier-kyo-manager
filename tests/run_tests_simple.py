#!/usr/bin/env python3
"""
簡単なテスト実行スクリプト
"""

import os
import sys

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """テストファイルのインポートを確認"""
    print("Testing imports...")

    try:
        from tests.test_plp_driver import *

        print("✓ test_plp_driver.py imported successfully")
    except Exception as e:
        print(f"✗ test_plp_driver.py import failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    try:
        from tests.test_product_extractor import *

        print("✓ test_product_extractor.py imported successfully")
    except Exception as e:
        print(f"✗ test_product_extractor.py import failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    try:
        from tests.test_browser_use_agent_plp_integration import *

        print("✓ test_browser_use_agent_plp_integration.py imported successfully")
    except Exception as e:
        print(f"✗ test_browser_use_agent_plp_integration.py import failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
