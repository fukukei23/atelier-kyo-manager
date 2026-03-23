#!/usr/bin/env python3
"""
SSENSE - 実際の商品URL探索
"""
import asyncio
import re
import sys
sys.path.insert(0, '/home/yn441611/atelier-kyo-manager')

async def find_product_urls():
    from playwright.async_api import async_playwright

    SSENSE_URL = "https://www.ssense.com/en-us/women/outerwear"
    print(f"SSENSE 商品URL探索: {SSENSE_URL}")
    print("=" * 70)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            locale="en-US",
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        )
        page = await context.new_page()

        try:
            await page.goto(SSENSE_URL, timeout=60000)
            await page.wait_for_timeout(5000)

            # ゆっくりスクロール
            for _ in range(5):
                await page.evaluate("window.scrollBy(0, 800)")
                await page.wait_for_timeout(1000)

            html = await page.content()

            # パ/AltnPatterns商品名パターンでURL抽出
            print("商品URLパターンマッチ:")

            #  方法1: /product/ を含むURL
            urls1 = re.findall(r'href="([^"]*product[^"]*)"', html, re.IGNORECASE)
            print(f"  href containing 'product': {len(urls1)}件")

            #  方法2: Palm Angels などのブランド名URL
            brand_urls = re.findall(r'href="([^"]*(?:palm-angels|gucci|prada|balenciaga)[^"]*)"', html, re.IGNORECASE)
            print(f"  ブランド名URL: {len(brand_urls)}件")

            # 重複去除
            all_urls = list(set(urls1 + brand_urls))
            print(f"\nユニークなURL数: {len(all_urls)}件")

            print("\n" + "=" * 70)
            print("サンプルURL (上位10件):")
            for i, url in enumerate(all_urls[:10], 1):
                print(f"{i}. {url}")

            # a[href] 全パターン也表示
            print("\n" + "=" * 70)
            print("a[href]全パターンからproduct探索:")
            all_hrefs = await page.evaluate("""
                () => {
                    const links = document.querySelectorAll('a[href]');
                    const productLinks = [];
                    links.forEach(a => {
                        const href = a.href;
                        // SSENSE product URLs を探す
                        if (href.match(/\\/product\\//) || href.match(/palm-angels/) || href.match(/gucci/) || href.match(/prada/)) {
                            productLinks.push(href);
                        }
                    });
                    return productLinks;
                }
            """)
            print(f"発見: {len(all_hrefs)}件")

            for i, url in enumerate(all_hrefs[:10], 1):
                print(f"{i}. {url}")

        except Exception as e:
            print(f"エラー: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

        await browser.close()

if __name__ == "__main__":
    asyncio.run(find_product_urls())
