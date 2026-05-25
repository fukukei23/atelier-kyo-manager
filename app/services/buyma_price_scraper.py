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

_BLOCK_RE = re.compile(
    r'syo_id="(\d+)"[^>]*?'
    r'syo_name="([^"]+?)"[^>]*?'
    r'brand_name="([^"]+?)"[^>]*?'
    r'price="(\d+)"',
)

_NOISE_WORDS = {
    "中古", "USED", "used", "新品", "未使用", "即発", "国内発送",
    "送料込", "関税込", "送料・関税込", "送料・関税込み",
    "直営店買付", "セール", "人気", "入手困難", "限定",
    "レディース", "メンズ", "Unisex", "100%",
    "☆", "★", "♪", "【", "】", "！", "！",
    "全色", "全カラー", "カラバリあり",
}

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

BUYMA_UNAVAILABLE_BRANDS = {"Gucci", "Ferragamo", "Valentino", "Chloe"}

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

_MODEL_NUMBER_RE = re.compile(r"\b([A-Z0-9]{5,12})\b")


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
    for brand in _BRAND_NORMALIZE:
        if brand in upper:
            return brand
    return upper


def _extract_model_numbers(name: str) -> set[str]:
    upper = name.upper()
    numbers: set[str] = set()
    for m in _MODEL_NUMBER_RE.finditer(upper):
        token = m.group(1)
        has_alpha = any(c.isalpha() for c in token)
        has_digit = any(c.isdigit() for c in token)
        if has_alpha and has_digit:
            numbers.add(token)
    return numbers


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

    official_models = _extract_model_numbers(official_name)
    buyma_models = _extract_model_numbers(buyma_name)
    model_bonus = 0.0
    if official_models and buyma_models:
        if official_models & buyma_models:
            model_bonus = 0.4

    official_tokens = _extract_tokens(official_name.upper())
    buyma_tokens = set(_extract_tokens(buyma_upper))

    if not official_tokens:
        return min(1.0, model_bonus)

    hits = sum(1 for t in official_tokens if t in buyma_tokens or t in buyma_upper)
    token_score = hits / len(official_tokens)

    seq_score = SequenceMatcher(
        None, official_name.upper(), buyma_upper
    ).ratio()

    return min(1.0, 0.6 * token_score + 0.4 * seq_score + model_bonus)


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
        "buyma_source": "buyma_search",
    }


class BuymaPriceSearcher:
    """BUYMA価格検索。ブラウザセッションを使い回す。"""

    def __init__(self, headless: bool = True, timeout: int = 60):
        self.headless = headless
        self.timeout = timeout
        self._pw = None
        self._browser = None

    def _ensure_browser(self):
        if self._browser:
            return
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)

    def close(self):
        if self._browser:
            try:
                self._browser.close()
            except Exception:
                pass
        if self._pw:
            try:
                self._pw.stop()
            except Exception:
                pass
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
        ctx = self._browser.new_context(
            viewport={"width": 1280, "height": 800},
            locale="ja-JP",
            timezone_id="Asia/Tokyo",
            user_agent=_CHROME_HEADERS["User-Agent"],
        )
        page = ctx.new_page()

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
                        time.sleep(2)

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
                        time.sleep(2 * (attempt + 1))

            logger.error(f"[buyma] All 3 attempts failed for {product_name}: {last_error}")
            raise last_error
        finally:
            ctx.close()

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

            time.sleep(2)

        return results


def search_buyma(
    query: str,
    brand: str | None = None,
    headless: bool = True,
    timeout: int = 60,
) -> list[dict]:
    encoded = urllib.parse.quote(query)
    url = f"https://www.buyma.com/r/{encoded}/"
    logger.info(f"[buyma] Searching: {url}")

    results: list[dict] = []
    try:
        from playwright.sync_api import sync_playwright
        from playwright_stealth import Stealth

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
            time.sleep(3)

            html = page.content()
            raw = _parse_buyma_results(html)

            if brand:
                brand_norm = _normalize_brand(brand)
                results = [
                    r for r in raw
                    if brand_norm in _normalize_brand(r["brand"])
                ]
            else:
                results = raw

            ctx.close()
            browser.close()

    except Exception as e:
        logger.error(f"[buyma] Search failed: {e}")

    return results


def fetch_buyma_prices(
    products: list[dict],
    headless: bool = True,
) -> dict[str, dict]:
    searcher = BuymaPriceSearcher(headless=headless)
    try:
        return searcher.search_batch(products)
    finally:
        searcher.close()


_shared_searcher: BuymaPriceSearcher | None = None
_last_activity: float = 0.0
_MAX_IDLE_SECONDS = 300.0


def get_shared_searcher() -> BuymaPriceSearcher:
    global _shared_searcher, _last_activity
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
