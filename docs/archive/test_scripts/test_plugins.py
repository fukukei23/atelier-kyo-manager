#!/usr/bin/env python3
"""
GUCCI/PRADA Plugin テスト
"""
import asyncio
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

async def test_plugin(name: str, plugin_class, test_url: str):
    """Plugin基本動作テスト"""
    try:
        from playwright.async_api import async_playwright

        plugin = plugin_class()
        logger.info(f"=== {name} Plugin Test ===")
        logger.info(f"Plugin loaded: {plugin.site}")
        logger.info(f"HARD_PLP_URL: {plugin._HARD_PLP_URL}")

        # before_navigateテスト
        corrected = plugin.before_navigate(test_url, None)
        logger.info(f"before_navigate: {test_url} -> {corrected}")

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled", "--no-sandbox"]
            )
            ctx = await browser.new_context(
                locale="en-US",
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = await ctx.new_page()

            try:
                await page.goto(corrected, wait_until="commit", timeout=30000)
                logger.info(f"Title: {await page.title()}")
            except Exception as e:
                logger.warning(f"goto warning: {e}")

            await plugin.after_navigate(page, None)

            success = await plugin.materialize(page, None)
            logger.info(f"materialize: {success}")

            plp_ok = await plugin.assert_plp(page, None)
            logger.info(f"assert_plp: {plp_ok}")

            product_links = await page.locator("a[href*='/product/']").all()
            logger.info(f"Product links: {len(product_links)}")

            for i, link in enumerate(product_links[:5], 1):
                href = await link.get_attribute("href")
                logger.info(f"  {i}: {href}")

            await browser.close()
            return len(product_links)

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback; traceback.print_exc()
        return 0

async def main():
    from app.agents.plugins.gucci_plp_v1 import GucciPLPStrategy
    from app.agents.plugins.prada_plp_v1 import PradaPLPStrategy

    gucci_count = await test_plugin("GUCCI", GucciPLPStrategy, "https://www.gucci.com/us/en/women/handbags")
    prada_count = await test_plugin("PRADA", PradaPLPStrategy, "https://www.prada.com/us/en/women/bags")

    print(f"\n{'='*50}")
    print(f"GUCCI: {gucci_count}件")
    print(f"PRADA: {prada_count}件")
    print(f"{'='*50}")

if __name__ == "__main__":
    asyncio.run(main())
