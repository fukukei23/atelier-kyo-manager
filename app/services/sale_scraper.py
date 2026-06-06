"""
YOOX / SSENSE セール価格スクレイパー

YOOX (yoox.com)      — curl_cffi (HTML静的)、USD
SSENSE (ssense.com)  — playwright-stealth (SPA)、USD

ブランドページをスクレイプし、SaleScraper.scrape() が返す dict に
actual_purchase_jpy / actual_purchase_source を自動設定する。

これは BrandPriceScraper の公式ブランドスクレイパーとは別物で、
「セール仕入れ価格」を取得的のが目的。
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from app.services.brand_price_scraper import (
    _CHROME_HEADERS,
    _fetch_with_cffi,
    _make_item,
)
from app.services.brand_price_scraper import BrandPriceScraper as _StealthScraper
from app.services.brand_price_scraper import _utcnow

logger = logging.getLogger(__name__)

SALE_SITES = ["yoox", "ssense"]

# ---------------------------------------------------------------------------
# Brand slug mappings
# ---------------------------------------------------------------------------

_YOOX_SLUGS: dict[str, str] = {
    "Gucci": "gucci",
    "Prada": "prada",
    "Valentino": "valentino-garavani",
    "Versace": "versace",
    "Marni": "marni",
    "Chloe": "chloe",
    "Celine": "celine",
    "Ferragamo": "salvatore-ferragamo",
    "Loewe": "loewe",
    "Balenciaga": "balenciaga",
    "Bottega Veneta": "bottega-veneta",
}

_SSENSE_SLUGS: dict[str, str] = {
    "Gucci": "gucci",
    "Prada": "prada",
    "Valentino": "valentino",
    "Versace": "versace",
    "Marni": "marni",
    "Chloe": "chloe",
    "Celine": "celine",
    "Ferragamo": "salvatore-ferragamo",
    "Loewe": "loewe",
    "Balenciaga": "balenciaga",
    "Bottega Veneta": "bottega-veneta",
}

# ---------------------------------------------------------------------------
# Bag-related category keywords (YOOX uses generic category names)
# ---------------------------------------------------------------------------

_BAG_KEYWORDS = {
    "handbag", "shoulder bag", "shoulderbags", "tote bag",
    "totes", "cross-body bag", "crossbody", "satchel",
    "clutch", "pouch", "mini bag", "chain bag", "bucket bag",
    "backpack", "travel bag", "messenger bag", "wallet",
    "bag", "bags",
}


def _slug_to_name(slug: str) -> str:
    """URL slug → readable name: 'red-rebelle-bag' → 'Red Rebelle Bag'."""
    words = slug.replace("-", " ").split()
    return " ".join(w.capitalize() for w in words if w)


def _is_bag(keyword: str) -> bool:
    kw = keyword.lower().strip()
    return any(k in kw for k in _BAG_KEYWORDS)


# ---------------------------------------------------------------------------
# YOOX scraper
# ---------------------------------------------------------------------------

def _scrape_yoox(brand: str, item_limit: int = 10) -> list[dict]:
    """
    YOOX women's bags brand page → list of sale items.

    URL: https://www.yoox.com/us/women/shoponline/{slug}_d
    HTML is server-rendered. Extracts:
      - Item codes from image URLs: /images/items/45/45886656nn_14_f.jpg
      - Category name from surrounding text
      - Sale price from "$2,001 (-36%)" pattern
    """
    slug = _YOOX_SLUGS.get(brand)
    if not slug:
        logger.warning(f"[yoox] No slug for brand {brand}")
        return []

    url = f"https://www.yoox.com/us/women/shoponline/{slug}_d"
    html = _fetch_with_cffi(url, timeout=30)
    if not html:
        logger.warning(f"[yoox] Failed to fetch {url}")
        return []

    results: list[dict] = []

    # YOOX item codes are embedded in image URLs
    # Pattern: /images/items/45/45886656NN_14_f.jpg → item code 45886656NN
    item_code_re = re.compile(r'/items/\d+/([A-Za-z0-9]+)_\d+_\w+\.jpg')
    # Price pattern: $ 2,001 (-36%)  or  $ 2,001
    price_re = re.compile(r'\$\s*([\d,]+)\s*(?:\(([-\d]+)%\))?')
    # Category name: appears before the item-code block
    # YOOX HTML: <span class="brand">GUCCI</span> ... <span>Handbags</span> ... price
    category_re = re.compile(
        r'<span[^>]*>\s*([A-Za-z][A-Za-z\s]{0,30}?(?:bag|handbag|tote|clutch|pouch|wallet|satchel)[A-Za-z\s]{0,30}?)\s*</span>',
        re.IGNORECASE,
    )

    # Split HTML around item code occurrences
    # For each item code, find the nearest category and price
    seen_codes: set[str] = set()
    for m in item_code_re.finditer(html):
        code = m.group(1).upper()
        if code in seen_codes:
            continue
        seen_codes.add(code)

        # Find surrounding context (500 chars before the match)
        start = max(0, m.start() - 500)
        context = html[start:m.end() + 200]

        # Extract category from context
        category_match = category_re.search(context)
        category = category_match.group(1).strip() if category_match else "Unknown"

        # Only keep bag categories
        if not _is_bag(category):
            continue

        # Extract price
        price_match = price_re.search(context)
        if not price_match:
            continue
        price = float(price_match.group(1).replace(",", ""))
        discount = int(price_match.group(2)) if price_match.group(2) else 0

        # Skip if not actually on sale (>10% off)
        if discount < 10:
            continue

        # Product name: category + item code
        product_name = f"{category.title()} #{code}"
        item_url = f"https://www.yoox.com/us/{code}/item"

        item = _make_item(brand, product_name, price, "yoox", item_url, currency="USD")
        item["actual_purchase_jpy"] = item["price_jpy"]
        item["actual_purchase_source"] = "yoox"
        item["discount_pct"] = discount
        results.append(item)

        if len(results) >= item_limit:
            break

    logger.info(f"[yoox] {brand}: {len(results)} sale items (out of {len(seen_codes)} found)")
    return results


# ---------------------------------------------------------------------------
# SSENSE scraper
# ---------------------------------------------------------------------------

def _scrape_ssense(brand: str, item_limit: int = 10) -> list[dict]:
    """
    SSENSE women's bags brand page → list of items with sale prices.

    URL: https://www.ssense.com/en-us/women/designers/{slug}/bags
    SPA with server-rendered metadata arrays in <head>.
    Extracts from page metadata (fast, no JS needed) + URL slugs for product names.
    """
    slug = _SSENSE_SLUGS.get(brand)
    if not slug:
        logger.warning(f"[ssense] No slug for brand {brand}")
        return []

    url = f"https://www.ssense.com/en-us/women/designers/{slug}/bags"

    # Try curl_cffi first (metadata is server-rendered even in SPA)
    html = _fetch_with_cffi(url, timeout=30)
    if not html:
        logger.warning(f"[ssense] curl_cffi blocked for {url}, trying playwright")
        # Fallback: playwright
        html = _ssense_via_playwright(url)
        if not html:
            return []

    results: list[dict] = []

    # SSENSE embeds product data as parallel arrays in metadata
    # Pattern: metadata = {url: [...], price: [...], sku: [...]}
    metadata_re = re.compile(
        r'metadata\s*=\s*(\{.*?\});',
        re.DOTALL,
    )
    m = metadata_re.search(html)
    if not m:
        # Fallback: parse from HTML structure
        return _ssense_parse_html_fallback(html, brand, url, item_limit)

    try:
        import json
        metadata = json.loads(m.group(1))
    except Exception:
        logger.warning("[ssense] Failed to parse metadata JSON")
        return _ssense_parse_html_fallback(html, brand, url, item_limit)

    urls: list[str] = metadata.get("url", [])
    prices_raw: list[Any] = metadata.get("price", [])
    skus: list[str] = metadata.get("sku", [])

    seen_products: set[str] = set()

    for i in range(min(len(urls), 100)):
        try:
            p_raw = prices_raw[i]
            # Price can be int/float or dict {"amount": ...}
            if isinstance(p_raw, dict):
                price = float(p_raw.get("amount", 0))
            else:
                price = float(p_raw)
        except (ValueError, TypeError, IndexError):
            continue

        if price <= 0:
            continue

        product_url = urls[i]
        if not product_url:
            continue

        # Extract name from URL slug
        # e.g. /women/product/gucci/red-rebelle-bag/3426109
        slug_match = re.search(r'/product/[^/]+/([^/]+)/\d+$', product_url)
        if slug_match:
            product_slug = slug_match.group(1)
            product_name = _slug_to_name(product_slug)
        else:
            product_name = f"Item {skus[i]}" if i < len(skus) else "Unknown"

        # Check if bag-related (filter out non-bag categories)
        name_lower = product_name.lower()
        if not _is_bag(name_lower) and not any(
            kw in name_lower for kw in ["dress", "top", "pant", "shoe", "boot", "jacket", "coat"]
        ):
            # If category is unclear, include it (SSENSE filters by bags/ URL already)
            pass

        if product_name in seen_products:
            continue
        seen_products.add(product_name)

        item = _make_item(brand, product_name, price, "ssense", product_url, currency="USD")
        item["actual_purchase_jpy"] = item["price_jpy"]
        item["actual_purchase_source"] = "ssense"
        results.append(item)

        if len(results) >= item_limit:
            break

    logger.info(f"[ssense] {brand}: {len(results)} items")
    return results


def _ssense_via_playwright(url: str) -> str | None:
    """Fallback: use playwright to render SSENSE SPA."""
    try:
        scraper = _StealthScraper(headless=True)
        scraper._ensure_browser()
        ctx = scraper._browser.new_context(
            viewport={"width": 1280, "height": 800},
            locale="en-US",
            user_agent=_CHROME_HEADERS["User-Agent"],
        )
        page = ctx.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        time.sleep(5)
        html = page.content()
        ctx.close()
        scraper.close()
        return html
    except Exception as e:
        logger.warning(f"[ssense] Playwright fallback failed: {e}")
        return None


def _ssense_parse_html_fallback(html: str, brand: str, url: str, item_limit: int) -> list[dict]:
    """Fallback: parse SSENSE HTML directly when metadata JSON unavailable."""
    results: list[dict] = []

    # Extract product links and prices from rendered HTML
    # Product cards: <a href="/en-us/women/product/..."> ... <span>$2,600</span>
    product_link_re = re.compile(
        r'href="(/en-us/women/product/[^"]+)"[^>]*>.*?<span[^>]*>\$?([\d,]+)',
        re.DOTALL,
    )
    price_re = re.compile(r'\$?([\d,]+)')

    seen: set[str] = set()
    for m in product_link_re.finditer(html):
        product_url = "https://www.ssense.com" + m.group(1)
        price_str = m.group(2).replace(",", "")
        try:
            price = float(price_str)
        except ValueError:
            continue

        slug_match = re.search(r'/product/[^/]+/([^/]+)/\d+$', product_url)
        if slug_match:
            product_name = _slug_to_name(slug_match.group(1))
        else:
            product_name = "Unknown"

        if product_name in seen:
            continue
        seen.add(product_name)

        item = _make_item(brand, product_name, price, "ssense", product_url, currency="USD")
        item["actual_purchase_jpy"] = item["price_jpy"]
        item["actual_purchase_source"] = "ssense"
        results.append(item)

        if len(results) >= item_limit:
            break

    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

class SaleScraper:
    """YOOX + SSENSE sale price scraper.

    scrape(brand) → list of dicts with actual_purchase_jpy set.
    """

    def scrape(
        self,
        brand: str,
        sites: list[str] | None = None,
        item_limit: int = 10,
    ) -> list[dict]:
        if sites is None:
            sites = SALE_SITES

        results: list[dict] = []

        if "yoox" in sites:
            results.extend(_scrape_yoox(brand, item_limit))

        if "ssense" in sites:
            results.extend(_scrape_ssense(brand, item_limit))

        logger.info(f"[SaleScraper] {brand}: {len(results)} total sale items")
        return results
