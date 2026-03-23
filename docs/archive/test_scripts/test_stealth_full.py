#!/usr/bin/env python3
"""
SSENSE Stealth 完全テスト
"""
import asyncio
import sys
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

async def main():
    try:
        from playwright.async_api import async_playwright
        from playwright_stealth.stealth import Stealth

        stealth = Stealth(
            navigator_webdriver=True,
            navigator_vendor=True,
            navigator_platform=True,
            navigator_languages=True,
            navigator_plugins=True,
            navigator_hardware_concurrency=True,
            chrome_load_times=True,
            chrome_csi=True,
            iframe_content_window=True,
            media_codecs=True,
            hairline=True,
            error_prototype=True,
            webgl_vendor=True,
            sec_ch_ua=True,
        )

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-blink-features",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--no-first-run",
                    "--no-zygote",
                    "--single-process",
                    "--ignore-certificate-errors",
                    "--disable-gpu",
                ]
            )
            ctx = await browser.new_context(
                locale="en-US",
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )

            # Apply stealth to context (covers all pages)
            await stealth.apply_stealth_async(ctx)

            page = await ctx.new_page()

            # Check webdriver flag
            wd = await page.evaluate("() => navigator.webdriver")
            logger.info(f"navigator.webdriver: {wd}")

            test_url = "https://www.ssense.com/en-us/women/outerwear"
            logger.info(f"Goto: {test_url}")

            await page.goto(test_url, wait_until="networkidle", timeout=60000)
            logger.info(f"Title: {await page.title()}")
            logger.info(f"URL: {page.url}")

            await asyncio.sleep(5)

            links = await page.locator("a[href*='/product/']").all()
            logger.info(f"\n商品リンク数: {len(links)}")
            for i, link in enumerate(links[:10], 1):
                href = await link.get_attribute("href")
                logger.info(f"  {i}: {href}")

            await browser.close()

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
