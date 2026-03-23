#!/usr/bin/env python3
"""
GUCCI PLP - スクレイピングテスト（代替URL）
"""
import asyncio
import json
import sys
sys.path.insert(0, '/home/yn441611/atelier-kyo-manager')

async def test_gucci():
    from playwright.async_api import async_playwright

    # 代替URLを試す
    GUCCI_URLS = [
        "https://www.gucci.com/ja/jp/c/womens-outerwear",
        "https://www.gucci.com/us/en/womens/ready-to-wear/outerwear",
        "https://www.gucci.com/us/en/womens",
    ]

    print(f"GUCCI スクレイピングテスト")
    print("=" * 70)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-dev-shm-usage']
        )
        context = await browser.new_context(
            locale="en-US",
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        )
        page = await context.new_page()

        for url in GUCCI_URLS:
            print(f"\n試行URL: {url}")
            try:
                response = await page.goto(url, timeout=30000, wait_until="domcontentloaded")
                await page.wait_for_timeout(3000)

                title = await page.title()
                print(f"タイトル: {title}")

                if "gucci" in title.lower():
                    print("GUCCIページを確認")

                    # 商品リンク
                    links = await page.evaluate("""
                        () => {
                            const l = [];
                            document.querySelectorAll('a[href*="/product/"]').forEach(a => l.push(a.href));
                            return [...new Set(l)];
                        }
                    """)
                    print(f"商品リンク数: {len(links)}件")

                    for i, u in enumerate(links[:5], 1):
                        print(f"  {i}. {u}")

                    if len(links) > 0:
                        print(f"\n成功: {len(links)}件の製品を発見")
                        break

            except Exception as e:
                print(f"エラー: {type(e).__name__}: {str(e)[:80]}")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(test_gucci())
