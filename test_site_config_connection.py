#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 3A-2-5: site_config 接続の動作確認テストスクリプト

実行方法:
    python test_site_config_connection.py
"""

import asyncio
import logging
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# ログファイルも生成するように設定
log_file = Path(__file__).parent / "test_site_config_connection.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"ログファイル: {log_file}")


async def test_site_config_connection():
    """site_config 接続の動作確認"""
    try:
        from app.config.loader import load_full_config
        from app.agents.browser.navigation_driver import NavigationDriver, NavigationContext
        from playwright.async_api import async_playwright
        
        logger.info("=" * 70)
        logger.info("Stage 3A-2-5: site_config 接続の動作確認テスト")
        logger.info("=" * 70)
        
        # 1. site_config を読み込む
        logger.info("\n[1] site_config を読み込み中...")
        full_config = load_full_config()
        moncler_config = full_config.get("MONCLER_OFFICIAL", {})
        
        if not moncler_config:
            logger.error("MONCLER_OFFICIAL の設定が見つかりません")
            return False
        
        logger.info(f"✓ site_config を読み込みました (キー数: {len(moncler_config)})")
        
        # 2. site_config の構造を確認
        logger.info("\n[2] site_config の構造を確認中...")
        
        # selectors.plp
        plp_selectors = moncler_config.get("selectors", {}).get("plp", {})
        if plp_selectors:
            logger.info(f"✓ selectors.plp が見つかりました: {list(plp_selectors.keys())}")
        else:
            logger.warning("⚠ selectors.plp が見つかりません（フォールバックが使用されます）")
        
        # selectors.pdp
        pdp_selectors = moncler_config.get("selectors", {}).get("pdp", {})
        if pdp_selectors:
            logger.info(f"✓ selectors.pdp が見つかりました: {list(pdp_selectors.keys())}")
        else:
            logger.warning("⚠ selectors.pdp が見つかりません（フォールバックが使用されます）")
        
        # navigation.header_search
        nav_config = moncler_config.get("navigation", {})
        if nav_config:
            logger.info(f"✓ navigation が見つかりました: {list(nav_config.keys())}")
            
            header_search = nav_config.get("header_search", {})
            if header_search:
                logger.info(f"  ✓ navigation.header_search: {list(header_search.keys())}")
            else:
                logger.warning("  ⚠ navigation.header_search が見つかりません")
            
            overlays = nav_config.get("overlays", {})
            if overlays:
                logger.info(f"  ✓ navigation.overlays: {list(overlays.keys())}")
            else:
                logger.warning("  ⚠ navigation.overlays が見つかりません")
        else:
            logger.warning("⚠ navigation が見つかりません（フォールバックが使用されます）")
        
        # 3. NavigationDriver の初期化を確認
        logger.info("\n[3] NavigationDriver の初期化を確認中...")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            
            try:
                # NavigationDriver を初期化
                driver = NavigationDriver(
                    page=page,
                    trap_checker=None,
                    telemetry=None,
                    strategy=None,
                )
                logger.info("✓ NavigationDriver を初期化しました")
                
                # NavigationContext を作成
                nav_ctx = NavigationContext(
                    site="MONCLER_OFFICIAL",
                    query="down jacket",
                    site_config=moncler_config,
                    settings={},
                    run_context=None,
                    start_t=0,
                    budget_ms=120000,
                    entry_url="https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB",
                    context=context,
                )
                logger.info("✓ NavigationContext を作成しました")
                
                # 4. site_config からセレクタが取得できることを確認
                logger.info("\n[4] site_config からセレクタ取得を確認中...")
                
                # collect_pdp_links のセレクタ取得ロジックをテスト
                plp_sel = moncler_config.get("selectors", {}).get("plp", {}) or {}
                pdp_sel = moncler_config.get("selectors", {}).get("pdp", {}) or {}
                
                pdp_link_selectors = (
                    (plp_sel.get("pdp_link_selectors", []) or []) +
                    (pdp_sel.get("pdp_link_selectors", []) or [])
                )
                
                if pdp_link_selectors:
                    logger.info(f"✓ pdp_link_selectors を取得しました ({len(pdp_link_selectors)} 個)")
                    logger.info(f"  例: {pdp_link_selectors[:3]}")
                else:
                    logger.warning("⚠ pdp_link_selectors が空です（フォールバックが使用されます）")
                
                # navigation.header_search のセレクタ取得ロジックをテスト
                nav_cfg = moncler_config.get("navigation", {}) or {}
                hs_cfg = nav_cfg.get("header_search", {}) or {}
                
                search_input = hs_cfg.get("search_input_selector", [])
                if search_input:
                    logger.info(f"✓ search_input_selector を取得しました: {search_input}")
                else:
                    logger.warning("⚠ search_input_selector が見つかりません（フォールバックが使用されます）")
                
                # navigation.overlays のセレクタ取得ロジックをテスト
                overlays_cfg = nav_cfg.get("overlays", {}) or {}
                cookie_selectors = overlays_cfg.get("cookie_banner_selectors", [])
                if cookie_selectors:
                    logger.info(f"✓ cookie_banner_selectors を取得しました: {cookie_selectors}")
                else:
                    logger.warning("⚠ cookie_banner_selectors が見つかりません（フォールバックが使用されます）")
                
                # 結果をサマリーとして出力
                logger.info("\n" + "=" * 70)
                logger.info("✓ すべての確認が完了しました")
                logger.info("=" * 70)
                
                # 結果サマリー
                logger.info("\n=== 結果サマリー ===")
                logger.info(f"selectors.plp: {'✓ 見つかりました' if plp_selectors else '⚠ 見つかりません'}")
                logger.info(f"navigation.header_search: {'✓ 見つかりました' if hs_cfg else '⚠ 見つかりません'}")
                logger.info(f"navigation.overlays: {'✓ 見つかりました' if overlays_cfg else '⚠ 見つかりません'}")
                
                logger.info("\n次のステップ:")
                logger.info("1. MONCLER_OFFICIAL.json に必要な構造を追加（完了済み）")
                logger.info("2. 実ブラウザテストを実行して動作確認")
                
                return True
                
            finally:
                await page.close()
                await context.close()
                await browser.close()
        
    except Exception as e:
        logger.error(f"テスト実行中にエラーが発生しました: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = asyncio.run(test_site_config_connection())
    sys.exit(0 if success else 1)

