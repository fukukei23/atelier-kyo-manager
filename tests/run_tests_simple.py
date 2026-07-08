#!/usr/bin/env python3
"""
簡単なテスト実行スクリプト
"""

import sys


def test_imports():
    """テストファイルのインポートを確認"""
    print("Testing imports...")

    try:
        import tests.test_plp_driver  # noqa: F401

        print("✓ test_plp_driver.py imported successfully")
    except Exception as e:
        print(f"✗ test_plp_driver.py import failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    try:
        import tests.test_product_extractor  # noqa: F401

        print("✓ test_product_extractor.py imported successfully")
    except Exception as e:
        print(f"✗ test_product_extractor.py import failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    try:
        import tests.test_browser_use_agent_plp_integration  # noqa: F401

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
