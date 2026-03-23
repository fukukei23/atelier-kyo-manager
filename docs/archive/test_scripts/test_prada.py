#!/usr/bin/env python3
"""
PRADA PLP - スクレイピングテスト
"""
import asyncio
import json
import sys
sys.path.insert(0, '/home/yn441611/atelier-kyo-manager')

async def test_prada():
    from playwright.async_api import async_playwright

    PRADA_URLS = [
        "https://www.prada.com/ww/en/womens/ready-to-wear/outerwear.html",
        "https://www.prada.com/us/en/womens/ready-to-wear.html",
    ]

    print(f"PRADA スクレイピングテスト")
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

        for url in PRADA_URLS:
            print(f"\n試行URL: {url}")
            try:
                response = await page.goto(url, timeout=30000, wait_until="domcontentloaded")
                await page.wait_for_timeout(3000)

                title = await page.title()
                print(f"タイトル: {title}")

                if "prada" in title.lower():
                    print("PRADAページを確認")

                    #  商品リンク
                    links = await page.evaluate("""
                        () => {
                            const l = [];
                            document.querySelectorAll('a[href*="/product/"], a[href*="/p/"]').forEach(a => l.push(a.href));
                            return [...new Set(l)];
                        }
                    """)
                    print(f"商品リンク数: {len(links)}件")

                    for i, u in enumerate(links[:5], 1):
                        print(f"  {i}. {u}")

                    # JSON-LD
                    scripts = await page.query_selector_all('script[type="application/ld+json"]')
                    prods = []
                    for script in scripts:
                        try:
                            content = await script.inner_text()
                            data = json.loads(content)
                            if isinstance(data, dict) and "@graph" in data:
                                prods.extend([x for x in data["@graph"] if x.get("@type") == "Product"])
                            elif data.get("@type") == "Product":
                                prods.append(data)
                        except:
                            pass
                    print(f"JSON-LD製品数: {len(prods)}件")

                    if len(links) > 0 or len(prods) > 0:
                        print(f"\n成功: {len(links)}件のリンク, {len(prods)}件のJSON-LD")
                        break

            except Exception as e:
                print(f"エラー: {type(e).__name__}: {str(e)[:80]}")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(test_prada())
