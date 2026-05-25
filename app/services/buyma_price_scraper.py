from __future__ import annotations

import logging
import re
import time
import urllib.parse
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

_CHROME_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}

_BUYMA_PRODUCT_RE = re.compile(
    r'syo_name="([^"]+?)"\s+brand_name="([^"]+?)"'
    r'[^>]*?price="(\d+)"',
)

# Noise words to strip from product names before matching
_NOISE_WORDS = {
    "中古", "USED", "used", "新品", "未使用", "即発", "国内発送",
    "送料込", "関税込", "送料・関税込", "送料・関税込み",
    "直営店買付", "セール", "人気", "入手困難", "限定",
    "レディース", "メンズ", "Unisex", "100%",
    "☆", "★", "♪", "【", "】", "！", "！",
    "全色", "全カラー", "カラバリあり",
}

# BUYMA brand name → normalized lookup
_BRAND_NORMALIZE = {
    "PRADA": "PRADA",
    "LOEWE": "LOEWE",
    "CELINE": "CELINE",
    "VERSACE": "VERSACE",
    "BALENCIAGA": "BALENCIAGA",
    "MARNI": "MARNI",
    "CHLOE": "CHLOE",
    "BOTTEGA VENETA": "BOTTEGA VENETA",
    "FERRAGAMO": "FERRAGAMO",
    "VALENTINO": "VALENTINO",
    "GUCCI": "GUCCI",
}


def _normalize_brand(name: str) -> str:
    upper = name.upper().replace("&AMP;", "&")
    for brand in _BRAND_NORMALIZE:
        if brand in upper:
            return brand
    return upper


def _extract_tokens(name: str) -> list[str]:
    clean = name
    for noise in _NOISE_WORDS:
        clean = clean.replace(noise, " ")
    clean = re.sub(r"[★☆♪【】!！・/／\s]+", " ", clean)
    tokens = [t for t in clean.split() if len(t) >= 2]
    return tokens


def _match_score(official_name: str, buyma_name: str, brand: str) -> float:
    buyma_upper = buyma_name.upper()
    brand_upper = brand.upper()

    if brand_upper not in buyma_upper:
        return 0.0

    official_tokens = _extract_tokens(official_name.upper())
    buyma_tokens = set(_extract_tokens(buyma_upper))

    if not official_tokens:
        return 0.0

    hits = sum(1 for t in official_tokens if t in buyma_tokens or t in buyma_upper)
    token_score = hits / len(official_tokens)

    seq_score = SequenceMatcher(
        None, official_name.upper(), buyma_upper
    ).ratio()

    return 0.6 * token_score + 0.4 * seq_score


def _parse_buyma_results(html: str) -> list[dict]:
    results: list[dict] = []
    seen_ids: set[str] = set()

    block_re = re.compile(
        r'syo_id="(\d+)"[^>]*?'
        r'syo_name="([^"]+?)"[^>]*?'
        r'brand_name="([^"]+?)"[^>]*?'
        r'price="(\d+)"',
    )

    for m in block_re.finditer(html):
        item_id = m.group(1)
        name = m.group(2)
        brand = m.group(3)
        price = m.group(4)

        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)

        results.append({
            "name": name.strip(),
            "brand": brand.replace("&amp;", "&"),
            "price_jpy": int(price),
            "item_id": item_id,
            "item_url": f"https://www.buyma.com/item/{item_id}/",
        })

    return results


def search_buyma(
    query: str,
    brand: str | None = None,
    headless: bool = True,
    timeout: int = 60,
) -> list[dict]:
    from playwright.sync_api import sync_playwright
    from playwright_stealth import Stealth

    encoded = urllib.parse.quote(query)
    url = f"https://www.buyma.com/r/{encoded}/"
    logger.info(f"[buyma] Searching: {url}")

    results: list[dict] = []
    try:
        with Stealth().use_sync(sync_playwright()) as p:
            browser = p.chromium.launch(headless=headless)
            ctx = browser.new_context(
                viewport={"width": 1280, "height": 800},
                locale="ja-JP",
                timezone_id="Asia/Tokyo",
                user_agent=_CHROME_HEADERS["User-Agent"],
            )
            page = ctx.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
            time.sleep(4)

            for _ in range(8):
                page.evaluate("window.scrollBy(0, 500)")
                time.sleep(0.3)

            html = page.content()
            raw = _parse_buyma_results(html)
            logger.info(f"[buyma] Raw results: {len(raw)}")

            if brand:
                brand_norm = _normalize_brand(brand)
                results = [
                    r for r in raw
                    if brand_norm in _normalize_brand(r["brand"])
                ]
            else:
                results = raw

            logger.info(f"[buyma] Filtered ({brand or 'all'}): {len(results)}")
            ctx.close()
            browser.close()

    except Exception as e:
        logger.error(f"[buyma] Search failed: {e}")

    return results


def match_product(
    official_name: str,
    brand: str,
    buyma_results: list[dict],
    threshold: float = 0.3,
) -> dict | None:
    if not buyma_results:
        return None

    scored: list[tuple[float, dict]] = []
    for r in buyma_results:
        score = _match_score(official_name, r["name"], brand)
        if score >= threshold:
            scored.append((score, r))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best = scored[0]

    same_name_items = [
        s[1] for s in scored
        if s[0] >= best_score * 0.9
    ]
    prices = [r["price_jpy"] for r in same_name_items]

    return {
        "buyma_name": best["name"],
        "buyma_price": min(prices),
        "buyma_price_max": max(prices),
        "buyma_price_avg": round(sum(prices) / len(prices)),
        "buyma_item_url": best["item_url"],
        "match_count": len(same_name_items),
        "match_score": round(best_score, 3),
        "buyma_source": "buyma_search",
    }


def fetch_buyma_prices(
    products: list[dict],
    headless: bool = True,
) -> dict[str, dict]:
    results: dict[str, dict] = {}

    for prod in products:
        name = prod.get("product_name", "")
        brand = prod.get("brand", "")
        if not name or not brand:
            continue

        query = f"{brand} {name}"
        buyma_items = search_buyma(query, brand=brand, headless=headless)
        match = match_product(name, brand, buyma_items)

        if match:
            results[name] = match
            logger.info(
                f"[buyma] Match: {name[:40]} → ¥{match['buyma_price']:,} "
                f"(score={match['match_score']}, {match['match_count']} items)"
            )
        else:
            logger.info(f"[buyma] No match: {name[:40]}")

        time.sleep(2)

    return results
