#!/usr/bin/env python3
"""
SSENSE Stealth テスト v5
"""
import asyncio
import sys
import logging
import json
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

async def main():
    try:
        from playwright.async_api import async_playwright
        from playwright_stealth.stealth import Stealth

        stealth_config = Stealth(navigator_webdriver=True, navigator_vendor=True,
                             navigator_platform=True, navigator_languages=True,
                             navigator_plugins=True, navigator_hardware_concurrency=True,
                             chrome_load_times=True, chrome_csi=True,
                             iframe_content_window=True, media_codecs=True,
                             hairline=True, error_prototype=True, webgl_vendor=True,
                             sec_ch_ua=True)

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox"
            ])
            ctx = await browser.new_context(
                locale="en-US",
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = await ctx.new_page()

            # Apply stealth directly to page
            stealth_config.apply_stealth_async(page)

            check = await page.evaluate("() => navigator.webdriver")
            logger.info(f"navigator.webdriver: {check}")

            test_url = "https://www.ssense.com/en-us/women/outerwear"
            logger.info(f"Goto: {test_url}")

            await page.goto(test_url, wait_until="networkidle", timeout=60000)
            logger.info(f"Title: {await page.title()}")
            logger.info(f"URL: {page.url}")

            await asyncio.sleep(5)

            links = await page.locator("a[href*='/product/']").all()
            logger.info(f"\n商品リンク数: {len(links)}")
            for i, link in enumerate(links[:20], 1):
                href = await link.get_attribute("href")
                logger.info(f"  {i}: {href}")

            for sel in ["div[class*='product']", "article"]:
                cnt = await page.locator(sel).count()
                if cnt > 0:
                    logger.info(f"  {sel}: {cnt}件")

            scripts = await page.query_selector_all('script[type="application/ld+json"]')
            logger.info(f"\nJSON-LD数: {len(scripts)}")
            for i, s in enumerate(scripts, 1):
                try:
                    c = await s.inner_text()
                    if c:
                        data = json.loads(c)
                        if isinstance(data, dict) and "@graph" in data:
                            prods = [x for x in data["@graph"] if x.get("@type") == "Product"]
                            logger.info(f"  Script {i}: @graph {len(prods)} products")
                            for pr in prods[:3]:
                                logger.info(f"    - {pr.get('name')}")
                        elif data.get("@type") == "Product":
                            logger.info(f"  Script {i}: Product - {data.get('name')}")
                except:
                    pass

            await browser.close()

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
