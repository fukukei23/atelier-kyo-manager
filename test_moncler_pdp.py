import asyncio
from playwright.async_api import async_playwright


async def apply_moncler_locale_patch(page):
    # locale と配送国を強制
    await page.add_init_script("""
        localStorage.setItem('client_locale', 'ja-JP');
    """)
    await page.context.add_cookies([{
        "name": "country",
        "value": "JP",
        "url": "https://www.moncler.com"
    }])


TARGET_URL = "https://www.moncler.com/ja-jp/men/outerwear/short-down-jackets/moncler-maya-hooded-short-down-jacket-olive-green-K20911A53600539ZD825.html"


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(
            locale="ja-JP",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )

        page = await context.new_page()

        # apply Moncler locale patch
        await apply_moncler_locale_patch(page)

        print("Navigating...")
        await page.goto(TARGET_URL, wait_until="domcontentloaded")

        # wait price area visible
        await page.wait_for_selector("text=¥", timeout=5000)

        print("Page Title:", await page.title())

        # Dump out __NEXT_DATA__ existence
        next_data = await page.evaluate("""() => {
            const el = document.querySelector("script#__NEXT_DATA__");
            return el ? el.textContent.slice(0, 500) : null;
        }""")
        print("\\nNEXT_DATA exists:", next_data is not None)

        # Grab DOM price text
        price_text = await page.inner_text("text=¥")
        print("DOM Price:", price_text)

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
