# ==============================================================================
# ファイル名 (File Name): product_info_extractor.py
# レジストリ (Registry): app/extractors/product_info_extractor.py
# 更新日時 (Date & Time JST): 2025-09-21 06:25:49
# バージョン (Version): 8.1.0J (Currency Data Enrichment)
#
# --- v8.1.0Jでの主な変更点 (あなたの最終レビューを反映) ---
# - [データリッチ化] Open Graphメタタグから価格を抽出する際に、
#   `product:price:currency` タグも同時に読み取り、通貨情報を
#   結果に含めるようにしました。
#
# --- v8.0.0Jからの維持機能 ---
# - JSON-LD → OGメタ → CSSの多段抽出ロジック。
# - 多通貨対応の数値抽出ヘルパー。
# ==============================================================================
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
import re
from typing import Any

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

logger = logging.getLogger(__name__)

VALID_PARSING_KEYS: set[str] = {"title", "price", "brand", "list_price", "discount_pct"}
CURRENCY_SYMBOLS = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "￥": "JPY",
    "₩": "KRW",
    "₽": "RUB",
    "₹": "INR",
    "₱": "PHP",
    "₺": "TRY",
    "₪": "ILS",
    "฿": "THB",
    "₫": "VND",
    "₦": "NGN",
    "₲": "PYG",
}


def _normalize_and_to_int(s: str | None) -> int | None:
    if not s:
        return None
    try:
        normalized = s.translate(str.maketrans("０１２３４5６７８９", "0123456789"))
        cleaned = re.sub(r"[^0-9.]", "", normalized)
        if not cleaned:
            return None
        return int(float(cleaned))
    except (ValueError, TypeError):
        return None


def _first_yen_from_text(text: str, preferred_currency: str | None = None) -> int | None:
    if not text:
        return None
    match = re.search(r"(?:¥|￥|\$|€)\s*([\d,.]+)|([\d,.]+)\s*(?:円|USD|EUR)", text, re.IGNORECASE)
    if match:
        return _normalize_and_to_int(match.group(1) or match.group(2))
    return _normalize_and_to_int(text)


def _norm_currency(cur: Any | None) -> str | None:
    """
    通貨表現を3桁コードへ正規化。シンボルも簡易対応。
    """
    if cur is None:
        return None
    try:
        c = str(cur).strip()
    except Exception:
        return None
    if not c:
        return None
    if c in CURRENCY_SYMBOLS:
        return CURRENCY_SYMBOLS[c]
    uc = c.upper()
    if len(uc) == 3:
        return uc
    # "usd", "Usd " など
    if uc in (
        "USD",
        "EUR",
        "GBP",
        "JPY",
        "CNY",
        "HKD",
        "SGD",
        "CAD",
        "AUD",
        "NZD",
        "CHF",
        "KRW",
        "TWD",
        "MXN",
        "BRL",
        "ARS",
        "TRY",
    ):
        return uc
    return None


def _parse_price_text(text: str) -> tuple[float | None, str | None]:
    """
    価格表記の文字列から (数値, 通貨コード) を推定する。
    - "¥ 123,000" / "USD 1200" / "1,299.99 EUR" などを簡易対応
    """
    if not text:
        return None, None
    cleaned = text.replace("\u00a0", " ").strip()
    # 通貨らしき部分と数値部分をゆるく抽出
    m = re.search(
        r"(?P<cur>[A-Z]{3}|USD|EUR|GBP|JPY|CNY|HKD|SGD|CAD|AUD|NZD|CHF|KRW|TWD|MXN|BRL|ARS|TRY|[€$£¥￥₩₽₹₱₺])?\s*(?P<amt>[0-9][0-9.,\s]+)",
        cleaned,
        re.IGNORECASE,
    )
    if not m:
        return None, None
    amt_raw = m.group("amt")
    if not amt_raw:
        return None, None
    # 桁区切り・空白を除去してから float 化
    amt_txt = amt_raw.replace(",", "").replace(" ", "")
    try:
        amt_val = float(amt_txt)
    except Exception:
        return None, None
    cur = _norm_currency(m.group("cur"))
    return amt_val, cur


def _flatten_dicts(obj: Any) -> list[dict[str, Any]]:
    """
    ネスト構造から dict ノードだけを抽出し、リスト化する。
    """
    found: list[dict[str, Any]] = []

    def _walk(o: Any) -> None:
        if isinstance(o, dict):
            found.append(o)
            for v in o.values():
                _walk(v)
        elif isinstance(o, list):
            for item in o:
                _walk(item)

    _walk(obj)
    return found


def _collect_ldjson(soup) -> list[dict[str, Any]]:
    """
    <script type="application/ld+json"> を収集し、JSONとして読み込む。
    """
    blobs: list[dict[str, Any]] = []
    for tag in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            txt = tag.string or tag.get_text() or ""
            if not txt.strip():
                continue
            data = json.loads(txt)
            blobs.extend(_flatten_dicts(data))
        except Exception:
            continue
    return blobs


def _extract_json_objects_from_text(text: str) -> list[Any]:
    """
    スクリプト文字列からJSONオブジェクトを抽出する。
    - window.__NEXT_DATA__ = {...};
    - const data = {...};
    - 生の JSON (配列/オブジェクト)
    """
    out: list[Any] = []
    if not text:
        return out
    stripped = text.strip()

    # 1) そのまま JSON として読める場合
    try:
        parsed = json.loads(stripped)
        out.append(parsed)
    except Exception:
        pass

    # 2) window.__NEXT_DATA__ = {...}; のようなパターン
    for m in re.finditer(r"=\s*({.*?})\s*;?", stripped, re.DOTALL):
        cand = m.group(1)
        try:
            parsed = json.loads(cand)
            out.append(parsed)
        except Exception:
            continue

    return out


def _collect_inline_json_candidates(soup) -> list[dict[str, Any]]:
    """
    __NEXT_DATA__ を含むあらゆる <script> 内の JSON をゆるく収集する。
    """
    blobs: list[dict[str, Any]] = []
    for tag in soup.find_all("script"):
        ttype = (tag.get("type") or "").lower()
        if "ld+json" in ttype:
            continue  # JSON-LD は別処理
        try:
            txt = tag.string or tag.get_text() or ""
            for obj in _extract_json_objects_from_text(txt):
                blobs.extend(_flatten_dicts(obj))
        except Exception:
            continue
    return blobs


def _dig_price_in_dict(obj: dict[str, Any]) -> tuple[float | None, str | None]:
    """
    ネスト辞書から price 相当の値を探索
    """
    # --- 1) 典型的なキー名（汎用） ----------------
    for key in ("price", "amount", "value", "salesPrice", "listPrice"):
        v = obj.get(key)
        if isinstance(v, (int, float)):
            return float(v), None
        if isinstance(v, str):
            amt, cur = _parse_price_text(v)
            if amt is not None:
                return amt, cur
        if isinstance(v, dict):
            # {"value": 123, "currency":"EUR"} / {"centAmount": 12300, "currencyCode": "EUR"}
            vv = v.get("value", v.get("amount", v.get("centAmount")))
            cc = v.get("currency", v.get("currencyCode"))
            if isinstance(vv, (int, float)):
                return (
                    float(vv) / 100.0 if key in ("centAmount",) or "centAmount" in v else float(vv)
                ), _norm_currency(cc)  # type: ignore
            if isinstance(vv, str):
                a, c = _parse_price_text(vv)
                return a, _norm_currency(cc) or c

    # --- 2) Moncler / 公式ブランドサイト向けのゆるい price 探索 -----------
    # window.__NEXT_DATA__ や embedded JSON 内で、
    # "formattedPrice", "currentPrice", "pricing" などのキーに価格が入ることがある。
    # ここでは「キー名に price / pricing を含むもの」を総当りする。
    for key, v in obj.items():
        lk = key.lower()
        # 既に上で処理した代表キーはスキップ
        if key in ("price", "amount", "value", "salesPrice", "listPrice"):
            continue
        # "price", "pricing", "formatted_price" などを対象にする
        if "price" not in lk and "pricing" not in lk:
            continue

        # パターンA: 文字列 "¥ 346,500" など
        if isinstance(v, str):
            amt, cur = _parse_price_text(v)
            if amt is not None:
                return amt, cur

        # パターンB: {"value": 346500, "currency": "JPY"} など
        if isinstance(v, dict):
            vv = v.get("value", v.get("amount", v.get("centAmount")))
            cc = v.get("currency", v.get("currencyCode"))
            if isinstance(vv, (int, float)):
                # centAmount の可能性を一応考慮
                val = float(vv)
                if "centamount" in (k.lower() for k in v):
                    val = val / 100.0
                return val, _norm_currency(cc)
            if isinstance(vv, str):
                a, c = _parse_price_text(vv)
                if a is not None:
                    return a, _norm_currency(cc) or c

        # パターンC: 数値単体だがキー名が "formattedPrice" など
        if isinstance(v, (int, float)):
            return float(v), None

    # --- 3) JSON-LD offers ------------------------------------------------
    if "offers" in obj:
        offers = obj["offers"]
        if isinstance(offers, dict):
            a = offers.get("price") or offers.get("priceSpecification", {}).get("price")  # type: ignore
            c = offers.get("priceCurrency")
            if isinstance(a, (int, float)):
                return float(a), _norm_currency(c)
            if isinstance(a, str):
                aa, cc = _parse_price_text(a)
                return aa, _norm_currency(c) or cc
        if isinstance(offers, list):
            for o in offers:
                if isinstance(o, dict):
                    a, c = _dig_price_in_dict(o)
                    if a is not None:
                        return a, c
    return None, None


def extract_product_info(html: str, site_config: dict[str, Any]) -> dict[str, Any]:
    if not BeautifulSoup:
        raise RuntimeError("BeautifulSoup4/lxml is not installed.")
    soup = BeautifulSoup(html, "lxml")
    pdp_selectors = site_config.get("selectors", {}).get("pdp", {})
    out: dict[str, Any] = {
        "title": None,
        "price": None,
        "currency": None,
        "brand": None,
        "list_price": None,
        "discount_pct": None,
        "source_flags": {},
        "raw": {},
    }

    # 1) タイトル（OG → <title> の順）
    if not out.get("title"):
        og_title = soup.select_one("meta[property='og:title']")
        if og_title and og_title.get("content"):
            out["title"] = og_title["content"]
            out["source_flags"]["title"] = "og:title"
    if not out.get("title") and soup.title:
        out["title"] = (soup.title.string or "").strip()

    # 2) ブランド（OG）
    if not out.get("brand"):
        og_brand = soup.select_one("meta[property='og:site_name']")
        if og_brand and og_brand.get("content"):
            out["brand"] = og_brand["content"]
            out["source_flags"]["brand"] = "og:site_name"

    # ---------- 3) 価格：DOM 可視テキスト ----------
    if out.get("price") is None:
        price_selectors = pdp_selectors.get("price") or [
            "[itemprop='price']",
            "[data-testid*='price' i]",
            "[class*='price' i]",
            "span.price, div.price, p.price",
        ]
        text_candidates: list[str] = []
        for sel in price_selectors:
            try:
                node = soup.select_one(sel)
                if node:
                    txt = node.get_text(" ", strip=True)
                    if txt:
                        text_candidates.append(txt)
            except Exception:
                continue

        for txt in text_candidates:
            amt, cur = _parse_price_text(txt)
            if amt is not None:
                out["price"] = amt
                if cur:
                    out["currency"] = cur
                out["raw"]["price_source"] = "dom_text"
                break

    # ---------- 4) メタタグ (OG/Schema) ----------
    if out.get("price") is None:
        meta_candidates = [
            ("meta[property='product:price:amount']", "content"),
            ("meta[itemprop='price']", "content"),
        ]
        for sel, attr in meta_candidates:
            try:
                tag = soup.select_one(sel)
                if tag:
                    val = tag.get(attr)
                    if val:
                        amt, cur = _parse_price_text(val)
                        if amt is not None:
                            out["price"] = amt
                            if cur:
                                out["currency"] = cur
                            out["raw"]["price_source"] = "meta"
                            out["source_flags"]["price"] = sel
                            og_currency = soup.select_one("meta[property='product:price:currency']")
                            if og_currency and og_currency.get("content"):
                                out["currency"] = _norm_currency(og_currency["content"]) or out.get("currency")
                                out["source_flags"]["currency"] = "og:currency"
                            break
            except Exception:
                continue

    # ---------- 5) JSON-LD ----------
    if out.get("price") is None:
        try:
            ldjson = _collect_ldjson(soup)
            for obj in ldjson:
                a, c = _dig_price_in_dict(obj)
                if a is not None:
                    out["price"] = a
                    if c:
                        out["currency"] = c
                    out["raw"]["price_source"] = "json_ld"
                    break
        except Exception:
            pass

    # ---------- 6) 埋め込み JSON（__NEXT_DATA__ 含む） ----------
    if out.get("price") is None:
        try:
            blobs = _collect_inline_json_candidates(soup)
            for obj in blobs:
                a, c = _dig_price_in_dict(obj)
                if a is not None:
                    out["price"] = a
                    if c:
                        out["currency"] = c
                    out["raw"]["price_source"] = "inline_json"
                    break
        except Exception:
            pass

    # ---------- 7) 最後の頼み：ページ全体テキスト走査 ----------
    if out.get("price") is None:
        try:
            full_text = soup.get_text(" ", strip=True)
            amt, cur = _parse_price_text(full_text or "")
            if amt is not None:
                out["price"] = amt
                if cur:
                    out["currency"] = cur
                out["raw"]["price_source"] = "body_text"
        except Exception:
            pass

    # (計算フォールバック、最終チェックは変更なし)
    return out


async def extract_title_price(page) -> dict[str, Any]:
    """
    BrowserUseAgent から利用されるタイトルと価格のシンプル抽出ヘルパー。
    Playwright Page オブジェクトを受け取り、title と価格テキストを返す。
    """
    title_text = ""
    price_text = ""
    currency = None

    try:
        title_text = (await page.title()) or ""
    except Exception:
        logger.debug("extract_title_price: page.title() 取得に失敗")

    try:
        price_node = await page.query_selector("p.price_color")
        if price_node:
            price_text = (await price_node.inner_text()) or ""
    except Exception:
        logger.debug("extract_title_price: price セレクタ取得に失敗")

    # 通貨推定（辞書マッチ）
    _CURRENCY_PREFIXES: list[tuple[str, str]] = [
        ("NZ$", "NZD"), ("NT$", "TWD"), ("MX$", "MXN"), ("HK$", "HKD"),
        ("S$", "SGD"), ("C$", "CAD"), ("CA$", "CAD"), ("A$", "AUD"),
        ("R$", "BRL"), ("AR$", "ARS"), ("CHF", "CHF"), ("Fr", "CHF"),
        ("kr", "SEK"), ("zł", "PLN"), ("Ft", "HUF"), ("Kč", "CZK"),
        ("د.إ", "AED"), ("﷼", "SAR"), ("ر.س", "SAR"),
    ]
    if price_text:
        stripped = price_text.strip()
        for prefix, code in _CURRENCY_PREFIXES:
            if stripped.startswith(prefix):
                currency = code
                break
        else:
            first = stripped[0] if stripped else ""
            currency = CURRENCY_SYMBOLS.get(first)

    return {
        "title": title_text.strip(),
        "price": price_text.strip(),
        "currency": currency,
        "url": page.url,
    }
