#!/usr/bin/env python3
import asyncio
import sys
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

async def main():
    try:
        from playwright.async_api import async_playwright
        test_url = "https://www.ssense.com/en-us/women/outerwear"

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            ctx = await browser.new_context(
                locale="en-US",
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = await ctx.new_page()

            # Log requests
            async def log_req(req):
                u = req.url.lower()
                if any(x in u for x in ["/product/", "graphql", "api"]):
                    logger.info(f"[REQ] {req.method} {req.url[:90]}")
            page.on("request", log_req)

            await page.goto(test_url, wait_until="networkidle", timeout=60000)
            logger.info(f"Title: {await page.title()}")
            logger.info(f"URL: {page.url}")

            await asyncio.sleep(5)

            links = await page.locator("a[href*='/product/']").all()
            logger.info(f"Product links: {len(links)}")
            for i, link in enumerate(links[:10], 1):
                href = await link.get_attribute("href")
                logger.info(f"  {i}: {href}")

            cnt = await page.locator("div.product-item").count()
            logger.info(f"div.product-item count: {cnt}")

            cnt2 = await page.locator("article").count()
            logger.info(f"article count: {cnt2}")

            scripts = await page.query_selector_all('script[type="application/ld+json"]')
            logger.info(f"JSON-LD scripts: {len(scripts)}")
            for i, s in enumerate(scripts, 1):
                c = await s.inner_text().catch(lambda _: "")
                if c:
                    logger.info(f"  Script {i}: len={len(c)}")
                    if "Product" in c:
                        logger.info(f"    -> Contains Product!")

            await browser.close()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
