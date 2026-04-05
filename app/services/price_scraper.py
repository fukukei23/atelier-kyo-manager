"""
F10: 価格・在庫スクレイピングサービス
source_urlから価格・在庫情報を自動取得する
"""

import json
import re
from datetime import datetime
from typing import Any, Dict, Optional

import requests
from bs4 import BeautifulSoup


class PriceScraper:
    """価格・在庫取得スクレイピングサービス"""

    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    TIMEOUT = 10

    STOCK_OUT_KEYWORDS = [
        "売切れ", "在庫切れ", "sold out", "out of stock",
        "在庫なし", "品切れ", "取り扱い終了",
        "入荷待ち", "comming soon", "back order",
    ]

    PRICE_PATTERNS = [
        r"¥\s*([0-9,]+)",
        r"€\s*([0-9,.]+)",
        r"\$\s*([0-9,.]+)",
        r"([0-9,]+)\s*円",
    ]

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ja,en-US;q=0.7,en;q=0.3",
        })

    def fetch(self, url: str) -> Dict[str, Any]:
        """URLから価格・在庫情報を取得"""
        result: Dict[str, Any] = {
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

    # ---- private -------------------------------------------------------
    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        tag = soup.find("title")
        if tag:
            return tag.get_text(strip=True)
        og = soup.find("meta", property="og:title")
        return og["content"] if og and og.get("content") else None

    def _extract_price(self, soup: BeautifulSoup, text: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {"price": None, "raw": None}

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
                price = (
                    offers.get("offers", {}).get("price")
                    or offers.get("price")
                )
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
            class_=lambda x: x and any(
                k in str(x).lower() for k in ["cart", "buy", "purchase"]
            ),
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
