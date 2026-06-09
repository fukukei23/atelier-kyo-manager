from __future__ import annotations

import logging
import re
import threading
import time
import urllib.parse

logger = logging.getLogger(__name__)

# Shared matching utilities (extracted to app/utils/product_matching.py)
from app.utils.product_matching import (
    NOISE_WORDS as _NOISE_WORDS,
    COLOR_NORMALIZE as _COLOR_NORMALIZE,
    MODEL_NUMBER_RE as _MODEL_NUMBER_RE,
    extract_model_numbers as _extract_model_numbers,
    extract_tokens as _extract_tokens,
    normalize_color as _normalize_color,
    match_score_buyma as _match_score,
)

_CHROME_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
}

_BLOCK_RE = re.compile(
    r'syo_id="(\d+)"[^>]*?'
    r'syo_name="([^"]+?)"[^>]*?'
    r'brand_name="([^"]+?)"[^>]*?'
    r'price="(\d+)"',
)

# 2026-06: BUYMA renewed HTML — item-url + brand-name-eigo + price attributes
_NEW_BLOCK_RE = re.compile(
    r'item-url="/item/(\d+)/?"[^>]*?'
    r'brand-name-eigo="([^"]*?)"[^>]*?'
    r'(?:price="(\d+)")?',
    re.DOTALL,
)

_BRAND_ALIASES: dict[str, str] = {
    "BOTTEGA VENETA": "BOTTEGA VENETA",
    "SALVATORE FERRAGAMO": "FERRAGAMO",
    "FERRAGAMO": "FERRAGAMO",
    "VALENTINO GARAVANI": "VALENTINO",
    "VALENTINO": "VALENTINO",
    "CHLOÉ": "CHLOE",
    "CHLOE": "CHLOE",
}

BUYMA_UNAVAILABLE_BRANDS = set()  # 2026-06: 全ブランド検索可能（旧: Gucci等をブロック）

BRAND_THRESHOLDS: dict[str, float] = {
    "Prada": 0.30,
    "Loewe": 0.30,
    "Celine": 0.25,
    "Versace": 0.30,
    "Balenciaga": 0.25,
    "Marni": 0.25,
    "Bottega Veneta": 0.25,
}
DEFAULT_THRESHOLD = 0.3
DEFAULT_TIMEOUT = 60
CACHE_DAYS = 7
SEARCH_THROTTLE = 2.0
PAGE_LOAD_WAIT = 2.0

_SIZE_RE = re.compile(r"\b(FREE|ONE\s*SIZE|ONE SIZE|F|XS|S|M|L|XL|XXL|2XL|3XL|(\d{2,3})\s*cm)\b", re.IGNORECASE)


def _build_search_query(product_name: str, brand: str) -> str:
    tokens = _extract_tokens(product_name)
    # 重要度順に最大5トークン + ブランド名
    scored = []
    for t in tokens:
        s = len(t)
        if any(c.isdigit() for c in t) and any(c.isalpha() for c in t):
            s += 10
        scored.append((s, t))
    scored.sort(reverse=True)
    top = [t for _, t in scored[:5]]
    return " ".join([brand] + top)


def _normalize_brand(name: str) -> str:
    upper = name.upper().replace("&AMP;", "&")
    for alias, canonical in _BRAND_ALIASES.items():
        if alias in upper:
            return canonical
    return upper


def _extract_size(name: str) -> str | None:
    m = _SIZE_RE.search(name)
    return m.group(0).strip() if m else None


def _filter_outliers(prices: list[int]) -> list[int]:
    if len(prices) < 4:
        return prices
    sorted_p = sorted(prices)
    q1 = sorted_p[len(sorted_p) // 4]
    q3 = sorted_p[3 * len(sorted_p) // 4]
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return [p for p in prices if lower <= p <= upper]


def _parse_buyma_results(html: str) -> list[dict]:
    results: list[dict] = []
    seen_ids: set[str] = set()

    # Try legacy format first (syo_id/syo_name/brand_name)
    for m in _BLOCK_RE.finditer(html):
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

    # If legacy found nothing, try new BUYMA format (2026-06: SPA renewed)
    if not results:
        # Extract from a[item-url] + price attribute
        _NEW_RE = re.compile(
            r'<a\b[^>]*?href="https?://www\.buyma\.com/item/(\d+)/?"[^>]*?'
            r'(?:brand-name-eigo="([^"]*)")?[^>]*?'
            r'(?:price="(\d+)")?[^>]*?>'
            r'(.*?)</a>',
            re.DOTALL,
        )
        for m in _NEW_RE.finditer(html):
            item_id = m.group(1)
            brand = m.group(2) or ""
            price_str = m.group(3)
            name_html = m.group(4) or ""

            if item_id in seen_ids:
                continue
            if not price_str:
                continue
            seen_ids.add(item_id)

            # Strip HTML tags from name
            name = re.sub(r"<[^>]+>", "", name_html).strip()
            if not name or len(name) < 3:
                continue

            results.append({
                "name": name,
                "brand": brand.replace("&amp;", "&"),
                "price_jpy": int(price_str),
                "item_id": item_id,
                "item_url": f"https://www.buyma.com/item/{item_id}/",
            })

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
    prices = _filter_outliers([r["price_jpy"] for r in same_name_items])

    return {
        "buyma_name": best["name"],
        "buyma_price": min(prices),
        "buyma_price_max": max(prices),
        "buyma_price_avg": round(sum(prices) / len(prices)),
        "buyma_item_url": best["item_url"],
        "match_count": len(same_name_items),
        "match_score": round(best_score, 3),
        "buyma_size": _extract_size(best["name"]),
        "buyma_source": "buyma_search",
    }


class BuymaPriceSearcher:
    """BUYMA価格検索。ブラウザセッションを使い回す。"""

    def __init__(self, headless: bool = True, timeout: int = DEFAULT_TIMEOUT):
        self.headless = headless
        self.timeout = timeout
        self._pw = None
        self._browser = None
        self._ctx = None

    def _ensure_browser(self):
        if self._browser:
            return
        from playwright.sync_api import sync_playwright
        from playwright_stealth import Stealth

        # playwright-stealth 2.x: wrap PlaywrightContextManager (not started Playwright)
        pw_cm = sync_playwright()
        self._stealth_cm = Stealth().use_sync(pw_cm)
        pw_instance = self._stealth_cm.__enter__()
        self._pw = pw_instance
        self._browser = self._pw.chromium.launch(headless=self.headless)

    def close(self):
        if self._ctx:
            try:
                self._ctx.close()
            except Exception:
                pass
        if self._browser:
            try:
                self._browser.close()
            except Exception:
                pass
        if hasattr(self, '_stealth_cm') and self._stealth_cm:
            try:
                self._stealth_cm.__exit__(None, None, None)
                self._stealth_cm = None
            except Exception:
                pass
        self._ctx = None
        self._browser = None
        self._pw = None

    def search_single(
        self,
        product_name: str,
        brand: str,
        threshold: float | None = None,
    ) -> dict | None:
        if brand in BUYMA_UNAVAILABLE_BRANDS:
            return None

        if threshold is None:
            threshold = BRAND_THRESHOLDS.get(brand, DEFAULT_THRESHOLD)

        self._ensure_browser()
        if not self._ctx:
            self._ctx = self._browser.new_context(
                viewport={"width": 1280, "height": 800},
                locale="ja-JP",
                timezone_id="Asia/Tokyo",
                user_agent=_CHROME_HEADERS["User-Agent"],
            )
        page = self._ctx.new_page()

        last_error = None
        try:
            for attempt in range(3):
                try:
                    query = urllib.parse.quote(_build_search_query(product_name, brand))
                    all_filtered = []
                    for pg in range(1, 3):
                        url = f"https://www.buyma.com/r/{query}/"
                        if pg > 1:
                            url = f"https://www.buyma.com/r/{query}/?page={pg}"
                        page.goto(url, wait_until="domcontentloaded", timeout=self.timeout * 1000)
                        time.sleep(PAGE_LOAD_WAIT)

                        html = page.content()
                        raw = _parse_buyma_results(html)
                        if not raw:
                            break

                        brand_norm = _normalize_brand(brand)
                        all_filtered.extend(
                            r for r in raw
                            if brand_norm in _normalize_brand(r["brand"])
                        )

                    return match_product(product_name, brand, all_filtered, threshold=threshold)
                except Exception as e:
                    last_error = e
                    logger.warning(f"[buyma] Attempt {attempt + 1}/3 failed for {product_name}: {e}")
                    if attempt < 2:
                        time.sleep(SEARCH_THROTTLE * (attempt + 1))

            logger.error(f"[buyma] All 3 attempts failed for {product_name}: {last_error}")
            raise last_error
        finally:
            page.close()

    def search_batch(
        self,
        products: list[dict],
    ) -> dict[str, dict]:
        results: dict[str, dict] = {}
        for prod in products:
            name = prod.get("product_name", "")
            brand = prod.get("brand", "")
            if not name or not brand:
                continue
            if brand in BUYMA_UNAVAILABLE_BRANDS:
                logger.info(f"[buyma] Skip unavailable brand: {brand} {name[:30]}")
                continue

            match = self.search_single(name, brand)
            if match:
                results[name] = match
                logger.info(
                    f"[buyma] Match: {name[:40]} -> Y{match['buyma_price']:,} "
                    f"(score={match['match_score']}, {match['match_count']} items)"
                )
            else:
                logger.info(f"[buyma] No match: {name[:40]}")

            time.sleep(SEARCH_THROTTLE)

        return results


_shared_searcher: BuymaPriceSearcher | None = None
_last_activity: float = 0.0
_MAX_IDLE_SECONDS = 300.0
_searcher_lock = threading.Lock()


def get_shared_searcher() -> BuymaPriceSearcher:
    global _shared_searcher, _last_activity
    with _searcher_lock:
        now = time.time()
        if _shared_searcher and (now - _last_activity) > _MAX_IDLE_SECONDS:
            try:
                _shared_searcher.close()
            except Exception:
                pass
            _shared_searcher = None
        if _shared_searcher is None:
            _shared_searcher = BuymaPriceSearcher(headless=True)
        _last_activity = now
        return _shared_searcher
