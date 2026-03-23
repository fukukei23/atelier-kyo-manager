#!/usr/bin/env python3
"""
SSENSE API直接取得テスト
"""
import asyncio
import sys
import logging
import json
import re
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

async def main():
    try:
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            ctx = await browser.new_context(
                locale="en-US",
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = await ctx.new_page()

            # Capture ALL responses
            api_responses = []
            async def log_response(resp):
                url = resp.url.lower()
                if any(x in url for x in ["product", "plp", "listing", "search", "catalog", "graphql"]):
                    try:
                        body = await resp.text()
                        logger.info(f"[API-RES] {resp.url[:100]}")
                        logger.info(f"  Status: {resp.status}, Size: {len(body)}")
                        if len(body) < 5000:
                            logger.info(f"  Body: {body[:500]}")
                        if "product" in url:
                            api_responses.append({"url": resp.url, "body": body})
                    except Exception as e:
                        logger.warning(f"  Failed to read: {e}")

            page.on("response", log_response)

            test_url = "https://www.ssense.com/en-us/women/outerwear"
            logger.info(f"Goto: {test_url}")

            await page.goto(test_url, wait_until="networkidle", timeout=60000)
            logger.info(f"Title: {await page.title()}")
            logger.info(f"URL: {page.url}")

            # スクロールして商品を読み込ませる
            logger.info("Scrolling...")
            for i in range(5):
                await page.evaluate(f"window.scrollBy(0, {500 + i * 100})")
                await asyncio.sleep(1)

            await asyncio.sleep(3)

            # 商品リンク確認
            links = await page.locator("a[href*='/product/']").all()
            logger.info(f"\n商品リンク数: {len(links)}")
            for i, link in enumerate(links[:10], 1):
                href = await link.get_attribute("href")
                logger.info(f"  {i}: {href}")

            # JSON-LD
            logger.info("\nJSON-LDスクリプト:")
            scripts = await page.query_selector_all('script[type="application/ld+json"]')
            for i, s in enumerate(scripts, 1):
                try:
                    c = await s.inner_text()
                    if c:
                        data = json.loads(c)
                        if isinstance(data, dict):
                            if "@graph" in data:
                                products = [x for x in data["@graph"] if x.get("@type") == "Product"]
                                logger.info(f"  Script {i}: @graph with {len(products)} products")
                                for prod in products[:3]:
                                    logger.info(f"    - {prod.get('name')} : {prod.get('url')}")
                            elif data.get("@type") == "Product":
                                logger.info(f"  Script {i}: Single Product - {data.get('name')}")
                        elif isinstance(data, list):
                            logger.info(f"  Script {i}: Array with {len(data)} items")
                except Exception as e:
                    logger.warning(f"  Script {i} error: {e}")

            # captured API responses
            logger.info(f"\nCaptured API responses: {len(api_responses)}")
            for r in api_responses[:3]:
                logger.info(f"  URL: {r['url'][:80]}")

            await browser.close()

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
