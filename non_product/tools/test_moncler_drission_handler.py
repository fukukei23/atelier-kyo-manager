#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MonclerDrissionHandler のテストスクリプト

実際にDrissionPageを使ってMONCLERサイトにアクセスし、動作確認を行います。
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.specialized.moncler_handler import MonclerDrissionHandler
from app.core.run_context import RunContext


def main():
    """MonclerDrissionHandler の動作確認"""
    
    print("=" * 80)
    print("MonclerDrissionHandler 動作確認")
    print("=" * 80)
    print()
    
    # DrissionPage が利用可能かチェック
    try:
        from DrissionPage import ChromiumPage
        print("✅ DrissionPage が利用可能です")
    except ImportError:
        print("❌ DrissionPage がインストールされていません")
        print("   以下のコマンドでインストールしてください:")
        print("   pip install DrissionPage")
        return 1
    
    # RunContext を作成
    run_context = RunContext()
    print(f"✅ RunContext 作成完了 (run_id: {run_context.run_id})")
    print()
    
    # サイト設定を読み込む
    try:
        from app.config.loader import load_and_merge_configs
        sites_config = load_and_merge_configs()
        moncler_config = sites_config.get("MONCLER_OFFICIAL")
        
        if not moncler_config:
            print("⚠️  MONCLER_OFFICIAL の設定が見つかりません")
            print("   デフォルト設定を使用します")
            moncler_config = {
                "base_url": "https://www.moncler.com",
                "navigation": {
                    "header_search": {
                        "search_input_selector": "input[name='q'], input[type='search']",
                        "submit_selector": "button[type='submit']",
                    }
                },
                "selectors": {
                    "plp": {
                        "pdp_link_selectors": [
                            "div[data-test='product-card'] a",
                            "li.product-grid__item a.product-tile__link",
                            "a[href*='/products/']",
                        ]
                    }
                }
            }
        else:
            print("✅ MONCLER_OFFICIAL の設定を読み込みました")
    except Exception as e:
        print(f"⚠️  設定読み込みエラー: {e}")
        print("   デフォルト設定を使用します")
        moncler_config = {
            "base_url": "https://www.moncler.com",
            "navigation": {
                "header_search": {
                    "search_input_selector": "input[name='q'], input[type='search']",
                }
            },
            "selectors": {
                "plp": {
                    "pdp_link_selectors": [
                        "a[href*='/products/']",
                    ]
                }
            }
        }
    
    print()
    
    # MonclerDrissionHandler を作成
    try:
        handler = MonclerDrissionHandler(
            runtime_kwargs={
                "headless": True,  # ヘッドレスモードで実行
            },
            user_data_path="user_data/moncler_profile_test",
        )
        print("✅ MonclerDrissionHandler 作成完了")
    except Exception as e:
        print(f"❌ MonclerDrissionHandler 作成エラー: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print()
    
    # テスト実行
    query = "down jacket"
    target_url = None  # 検索を使用
    
    print(f"クエリ: {query}")
    print(f"target_url: {target_url or '(検索を使用)'}")
    print()
    print("テスト実行中...")
    print()
    
    try:
        # 実行（同期関数として実行）
        result = handler.run(
            query=query,
            site_config=moncler_config,
            run_context=run_context,
            target_url=target_url,
        )
        
        print()
        print("=" * 80)
        print("実行結果")
        print("=" * 80)
        print(f"成功: {result.ok}")
        print(f"メッセージ: {result.message}")
        print(f"サイト: {result.site}")
        print(f"クエリ: {result.query}")
        
        if result.evidence and "extracted_data" in result.evidence:
            items = result.evidence["extracted_data"]
            print(f"取得した商品数: {len(items)}")
            
            if items:
                print()
                print("取得した商品（最初の3件）:")
                for i, item in enumerate(items[:3], 1):
                    print(f"  {i}. {item.get('title', 'N/A')}")
                    print(f"     価格: {item.get('price', 'N/A')}")
                    print(f"     URL: {item.get('url', 'N/A')}")
        
        print()
        print("=" * 80)
        
        if result.ok:
            print("✅ テスト成功")
            return 0
        else:
            print("❌ テスト失敗")
            return 1
            
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ テスト実行エラー")
        print("=" * 80)
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

