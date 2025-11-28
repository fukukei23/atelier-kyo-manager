#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cursor設定ファイル確認スクリプトを実行
"""
import sys
import importlib.util
from pathlib import Path

# スクリプトをインポートして実行
tools_dir = Path(__file__).parent / "tools"
script_path = tools_dir / "check_cursor_config.py"

try:
    spec = importlib.util.spec_from_file_location("check_cursor_config", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if hasattr(module, 'main'):
        module.main()
    else:
        print("❌ main関数が見つかりません")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ 実行エラー: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

