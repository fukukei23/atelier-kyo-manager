"""
F10: 価格・在庫スクレイピングサービス
source_urlから価格・在庫情報を自動取得する
"""

import json
import logging
import re
import time
from collections import OrderedDict
from typing import Any

import requests
from bs4 import BeautifulSoup

from app.config.constants import SCRAPER_CACHE_SIZE, SCRAPER_MAX_RETRIES, SCRAPER_REQUEST_TIMEOUT
from app.utils.validation import ValidationError, validate_url

logger = logging.getLogger(__name__)


class PriceScraper:
    """価格・在庫取得スクレイピングサービス"""

    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    TIMEOUT = SCRAPER_REQUEST_TIMEOUT

    STOCK_OUT_KEYWORDS = [
        "売切れ",
        "在庫切れ",
        "sold out",
        "out of stock",
        "在庫なし",
        "品切れ",
        "取り扱い終了",
        "入荷待ち",
        "comming soon",
        "back order",
    ]

    PRICE_PATTERNS = [
        r"¥\s*([0-9,]+)",
        r"€\s*([0-9,.]+)",
        r"\$\s*([0-9,.]+)",
        r"([0-9,]+)\s*円",
    ]

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "ja,en-US;q=0.7,en;q=0.3",
            }
        )
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def fetch(self, url: str) -> dict[str, Any]:
        """URLから価格・在庫情報を取得"""
        try:
            url = validate_url(url)
        except ValidationError as e:
            return {
                "success": False,
                "title": None,
                "price": None,
                "in_stock": True,
                "raw_price": None,
                "error": str(e),
            }

        result: dict[str, Any] = {
            "success": False,
            "title": None,
            "price": None,
            "in_stock": True,
            "raw_price": None,
            "error": None,
        }

        try:
            resp = self.session.get(url, timeout=self.TIMEOUT, allow_redirects=True)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"

            soup = BeautifulSoup(resp.text, "html.parser")
            result["title"] = self._extract_title(soup)
            price_info = self._extract_price(soup, resp.text)
            result["price"] = price_info["price"]
            result["raw_price"] = price_info["raw"]
            result["in_stock"] = self._check_stock(soup, resp.text)
            result["success"] = True

        except requests.Timeout:
            result["error"] = "タイムアウト"
        except requests.ConnectionError:
            result["error"] = "接続失敗"
        except requests.HTTPError as e:
            result["error"] = f"HTTP {e.response.status_code}"
        except Exception as e:
            result["error"] = str(e)

        return result

    # ---- FR-004: フォールバック機能 ---------------------------------------

    @staticmethod
    def classify_error(error_string: str) -> str:
        """エラー文字列からカテゴリを分類"""
        s = str(error_string)
        if any(k in s for k in ("403", "429", "Cloudflare", "cloudflare")):
            return "blocked"
        if "404" in s:
            return "not_found"
        if "タイムアウト" in s or "Timeout" in s:
            return "timeout"
        if "接続失敗" in s or "ConnectionError" in s:
            return "connection"
        if "HTTP 5" in s:
            return "server_error"
        return "unknown"

    def fetch_with_retry(self, url: str, max_retries: int = SCRAPER_MAX_RETRIES) -> dict[str, Any]:
        """リトライ付き取得（blocked/server_errorのみリトライ）"""
        attempts: list[dict[str, Any]] = []
        retry_count = 0
        result: dict[str, Any] = {}

        for attempt in range(max_retries):
            result = self.fetch(url)

            if result.get("success"):
                result["attempts"] = attempts
                result["retry_count"] = retry_count
                result["error_category"] = None
                return result

            error = result.get("error", "")
            error_category = self.classify_error(str(error))
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "error": error,
                    "error_category": error_category,
                }
            )

            # blocked/server_error以外はリトライしない
            if error_category not in ("blocked", "server_error"):
                result["attempts"] = attempts
                result["retry_count"] = retry_count
                result["error_category"] = error_category
                return result

            if attempt < max_retries - 1:
                retry_count += 1
                time.sleep(2**attempt)  # 指数バックオフ: 1s, 2s, 4s

        result["attempts"] = attempts
        result["retry_count"] = retry_count
        result["error_category"] = self.classify_error(str(result.get("error", "")))
        return result

    def fetch_cached(self, url: str, cache_hours: int = 24) -> dict[str, Any]:
        """キャッシュ付き取得（同一URL 24h以内は再利用）"""
        if url in self._cache:
            cached = self._cache[url]
            if time.time() - cached["timestamp"] < cache_hours * 3600:
                return cached["result"]

        result = self.fetch_with_retry(url)
        self._cache[url] = {"result": result, "timestamp": time.time()}
        if len(self._cache) > SCRAPER_CACHE_SIZE:
            self._cache.popitem(last=False)
        return result

    # ---- private -------------------------------------------------------
    def _extract_title(self, soup: BeautifulSoup) -> str | None:
        tag = soup.find("title")
        if tag:
            return tag.get_text(strip=True)
        og = soup.find("meta", property="og:title")
        return og["content"] if og and og.get("content") else None

    def _extract_price(self, soup: BeautifulSoup, text: str) -> dict[str, Any]:
        result: dict[str, Any] = {"price": None, "raw": None}

        # og:price:amount
        og = soup.find("meta", property="og:price:amount")
        if og and og.get("content"):
            try:
                result["price"] = int(float(og["content"]))
                result["raw"] = og["content"]
                return result
            except (ValueError, TypeError):
                pass

        # JSON-LD
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
                offers = data if isinstance(data, dict) else {}
                price = offers.get("offers", {}).get("price") or offers.get("price")
                if price:
                    result["price"] = int(float(str(price)))
                    result["raw"] = str(price)
                    return result
            except (json.JSONDecodeError, TypeError, KeyError, AttributeError):
                continue

        # 正規表現フォールバック
        for pat in self.PRICE_PATTERNS:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                raw = m.group(0)
                result["raw"] = raw
                nums = re.sub(r"[^\d.]", "", m.group(1))
                if nums:
                    try:
                        result["price"] = int(float(nums))
                        return result
                    except ValueError:
                        pass
        return result

    def _check_stock(self, soup: BeautifulSoup, text: str) -> bool:
        lower = text.lower()
        for kw in self.STOCK_OUT_KEYWORDS:
            if kw.lower() in lower:
                return False

        # カートボタン確認
        btn = soup.find(
            ["button", "a", "input"],
            class_=lambda x: x and any(k in str(x).lower() for k in ["cart", "buy", "purchase"]),
        )
        if btn:
            if btn.get("disabled") is not None:
                return False
            btn_text = btn.get_text(strip=True).lower()
            if any(k in btn_text for k in ["sold out", "売切れ", "在庫切れ"]):
                return False
        return True

    def close(self) -> None:
        self.session.close()
