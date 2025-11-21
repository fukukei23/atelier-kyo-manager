# ==============================================================================
# ファイル名: app/extractors/product_info_extractor.py
# 日付: 2025年10月25日 (日本時間)
#
# メモ:
# これは `product_info_extractor.py` の完全版（全置換用コード）です。
#
# 目的:
# 既存の product_info_extractor.py の機能（extract_product_info）をすべて維持しつつ、
# browser_use_agent.py が期待している旧API `extract_title_price` を互換ラッパー
# として末尾に追加し、ImportError を解決します。
#
# 使用方法:
# 以下のコード全体をコピーし、既存の `app/extractors/product_info_extractor.py`
# の内容と完全に置き換えてください。
# ==============================================================================

from __future__ import annotations

import asyncio
import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from playwright.async_api import Page, TimeoutError as PWTimeout

# ============== 正規化ユーティリティ ==============

_CURRENCY_SYMBOLS: Dict[str, str] = {
    "¥": "JPY",
    "￥": "JPY",
    "円": "JPY",
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "₩": "KRW",
    "₺": "TRY",
    "₫": "VND",
    "₱": "PHP",
    "₽": "RUB",
    "Fr": "CHF",
}

_CURRENCY_CODES = {
    "JPY", "USD", "EUR", "GBP", "KRW", "CNY", "TWD", "HKD", "AUD", "CAD", "CHF",
    "SGD", "THB", "VND", "TRY", "PHP", "RUB", "SEK", "NOK", "DKK", "PLN"
}

_PRICE_PAT = re.compile(
    r"(?P<cur>(?:USD|EUR|GBP|JPY|KRW|CNY|TWD|HKD|AUD|CAD|CHF|SGD|THB|VND|TRY|PHP|RUB|SEK|NOK|DKK|PLN|[$€£¥￥]))\s*"
    r"(?P<num>(?:\d{1,3}(?:[.,]\d{3})*|\d+)(?:[.,]\d{2})?)"
    r"|(?P<num_only>(?:\d{1,3}(?:[.,]\d{3})*|\d+)(?:[.,]\d{2})?)\s*(?P<cur_after>(?:USD|EUR|GBP|JPY|KRW|CNY|TWD|HKD|AUD|CAD|CHF|SGD|THB|VND|TRY|PHP|RUB|SEK|NOK|DKK|PLN))",
    re.IGNORECASE
)

def _norm_currency(cur: Optional[str]) -> Optional[str]:
    if not cur:
        return None
    cur = cur.strip()
    if cur in _CURRENCY_SYMBOLS:
        return _CURRENCY_SYMBOLS[cur]
    up = cur.upper()
    return up if up in _CURRENCY_CODES else None

def _norm_amount(raw: str) -> Optional[float]:
    """
    '1.234,00' -> 1234.00（EU型）
    '1,234.00' -> 1234.00（US型）
    '1234'     -> 1234.0
    """
    s = raw.strip()
    # どちらの小数記号かを推定
    if s.count(",") and s.count("."):
        # 末尾2桁の前にある区切りを小数点とみなす
        if re.search(r"[.,]\d{2}$", s):
            dec = re.search(r"([.,])\d{2}$", s).group(1)  # type: ignore
            thou = "," if dec == "." else "."
            s = s.replace(thou, "")
            s = s.replace(dec, ".")
        else:
            s = s.replace(",", "")
    else:
        # 片方だけ
        if s.count(",") and not s.count("."):
            # 多くは欧州表記: カンマが小数点の役割
            if re.search(r",\d{2}$", s):
                s = s.replace(".", "")
                s = s.replace(",", ".")
            else:
                s = s.replace(",", "")
        else:
            s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return None

def _parse_price_text(txt: str) -> Tuple[Optional[float], Optional[str]]:
    """
    テキストから (amount, currency) を抽出
    """
    for m in _PRICE_PAT.finditer(txt):
        cur = None
        num = None
        if m.group("cur") and m.group("num"):
            cur = _norm_currency(m.group("cur"))
            num = _norm_amount(m.group("num"))
        elif m.group("num_only") and m.group("cur_after"):
            cur = _norm_currency(m.group("cur_after"))
            num = _norm_amount(m.group("num_only"))
        if num is not None:
            return num, cur
    return None, None

# ============== DOM 抽出ヘルパー ==============

async def _inner_texts(page: Page, selectors: List[str]) -> List[str]:
    js = """
    (sels) => {
      const out = [];
      for (const sel of sels) {
        for (const el of document.querySelectorAll(sel)) {
          const t = (el.innerText || el.textContent || "").trim();
          if (t) out.push(t);
        }
      }
      return out;
    }
    """
    try:
        arr = await page.evaluate(js, selectors)  # type: ignore[arg-type]
        return [a for a in arr if isinstance(a, str)]
    except Exception:
        return []

async def _attr_contents(page: Page, selector: str, attr: str) -> List[str]:
    js = """
    (sel, attr) => {
      const out = [];
      for (const el of document.querySelectorAll(sel)) {
        const v = el.getAttribute(attr);
        if (v && v.trim()) out.push(v.trim());
      }
      return out;
    }
    """
    try:
        arr = await page.evaluate(js, selector, attr)  # type: ignore[arg-type]
        return [a for a in arr if isinstance(a, str)]
    except Exception:
        return []

async def _collect_ldjson(page: Page) -> List[Dict[str, Any]]:
    js = """
    () => {
      const out = [];
      const nodes = document.querySelectorAll('script[type="application/ld+json"]');
      for (const n of nodes) {
        try {
          const txt = n.textContent || n.innerText || "";
          if (!txt.trim()) continue;
          const obj = JSON.parse(txt);
          if (Array.isArray(obj)) { obj.forEach(o => out.push(o)); }
          else { out.push(obj); }
        } catch(e) {}
      }
      return out;
    }
    """
    try:
        data = await page.evaluate(js)  # type: ignore
        return [d for d in data if isinstance(d, dict)]
    except Exception:
        return []

async def _collect_inline_json_candidates(page: Page) -> List[Dict[str, Any]]:
    """
    __NEXT_DATA__ や任意の script 内に潜む price フィールドを総当り。
    JSON としてパース可能なブロックを抽出し、price/amount/value を持つ辞書を返す。
    """
    js = """
    () => {
      const out = [];

      // 1) __NEXT_DATA__ や data-json の塊
      const nodes = document.querySelectorAll('script');
      for (const n of nodes) {
        const id = n.getAttribute('id') || '';
        const t  = n.textContent || n.innerText || '';
        if (!t || !t.trim()) continue;

        // 素直に JSON のとき
        if (t.trim().startsWith('{') || t.trim().startsWith('[')) {
          try {
            const obj = JSON.parse(t);
            out.push(obj);
            continue;
          } catch(e) {}
        }

        // 非 JSON テキストから price を含む JSON らしき塊を緩く抽出
        // {"...price":...} のようなブロックを拾う
        try {
          const candidates = [];
          const rx = /\\{[^]*?\\}/g; // 大括弧の最短貪欲。サイズ大のときは Playwright 側で弾かれる
          let m;
          while ((m = rx.exec(t)) !== null) {
            const s = m[0];
            if (/"price"|\\bamount\\b|\\bvalue\\b/.test(s)) {
              candidates.push(s);
            }
          }
          for (const c of candidates) {
            try {
              const obj = JSON.parse(c);
              out.push(obj);
            } catch(e) {}
          }
        } catch(e) {}
      }

      // window.__NEXT_DATA__ 参照
      try {
        if (window.__NEXT_DATA__) out.push(window.__NEXT_DATA__);
      } catch(e) {}

      return out;
    }
    """
    try:
        data = await page.evaluate(js)  # type: ignore
        # フラット化して dict のみ返す
        out: List[Dict[str, Any]] = []

        def _walk(o: Any):
            if isinstance(o, dict):
                out.append(o)
                for v in o.values():
                    _walk(v)
            elif isinstance(o, list):
                for v in o:
                    _walk(v)

        for root in data:
            _walk(root)
        # サイズ制御（念のため）
        return out[:2000]
    except Exception:
        return []

def _dig_price_in_dict(obj: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    """
    ネスト辞書から price 相当の値を探索
    """
    # よくあるパターン
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
                return (float(vv) / 100.0 if key in ("centAmount",) or "centAmount" in v else float(vv)), _norm_currency(cc)  # type: ignore
            if isinstance(vv, str):
                a, c = _parse_price_text(vv)
                return a, _norm_currency(cc) or c
    # JSON-LD offers
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

# ============== 公開 API ==============

async def extract_product_info(page: Page, site: str = "", site_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    単一 PDP を想定した抽出器。
    返り値: {"name", "price", "currency", "url", "images"} など。
    価格は複数経路で探索し、最初に成立したものを返す。
    """
    result: Dict[str, Any] = {
        "url": page.url,
        "site": site,
        "name": None,
        "price": None,
        "currency": None,
        "images": [],
        "raw": {},
    }

    # ---------- 1) タイトル/商品名 ----------
    try:
        name_candidates = await _inner_texts(page, [
            "h1",
            "[data-testid*='product'] h1",
            "[itemprop='name']",
            "meta[property='og:title']",
        ])
        if name_candidates:
            # og:title が入ってきた時にテキストと被るので最長を採用
            result["name"] = max(name_candidates, key=len)
    except Exception:
        pass

    # ---------- 2) 画像 ----------
    try:
        imgs = await _attr_contents(page, "img[src]", "src")
        imgs += await _attr_contents(page, "img[data-src]", "data-src")
        imgs = list(dict.fromkeys(imgs))  # 重複排除
        result["images"] = imgs[:20]
    except Exception:
        pass

    # ---------- 3) 価格：DOM 可視テキスト ----------
    text_candidates = await _inner_texts(page, [
        # まずは一般的な価格表示
        "[itemprop='price']",
        "[data-testid*='price' i]",
        "[class*='price' i]",
        "[class*='Price' i]",
        "span:has([itemprop='price'])",
    ])
    # Moncler っぽい補助（将来名称変更に強めのゆるマッチ）
    text_candidates += await _inner_texts(page, [
        "[data-test*='price' i]",
        "div[class*='pdp'] [class*='price' i]",
    ])
    for txt in text_candidates:
        amt, cur = _parse_price_text(txt)
        if amt is not None:
            result["price"] = amt
            if cur:
                result["currency"] = cur
            break

    # ---------- 4) meta / itemprop ----------
    if result["price"] is None:
        try:
            meta_price = await _attr_contents(page, "meta[itemprop='price']", "content")
            meta_cur = await _attr_contents(page, "meta[itemprop='priceCurrency']", "content")
            if meta_price:
                amt = _norm_amount(meta_price[0])
                if amt is not None:
                    result["price"] = amt
                    if meta_cur:
                        result["currency"] = _norm_currency(meta_cur[0]) or result["currency"]
        except Exception:
            pass

    # ---------- 5) JSON-LD ----------
    if result["price"] is None:
        try:
            ldjson = await _collect_ldjson(page)
            for obj in ldjson:
                a, c = _dig_price_in_dict(obj)
                if a is not None:
                    result["price"] = a
                    if c:
                        result["currency"] = c
                    break
        except Exception:
            pass

    # ---------- 6) 埋め込み JSON（__NEXT_DATA__ 含む） ----------
    if result["price"] is None:
        try:
            blobs = await _collect_inline_json_candidates(page)
            for obj in blobs:
                a, c = _dig_price_in_dict(obj)
                if a is not None:
                    result["price"] = a
                    if c:
                        result["currency"] = c
                    break
        except Exception:
            pass

    # ---------- 7) 最後の頼み：ページ全体テキスト走査 ----------
    if result["price"] is None:
        try:
            full_text = await page.evaluate("() => document.body ? document.body.innerText || '' : ''")
            amt, cur = _parse_price_text(full_text or "")
            if amt is not None:
                result["price"] = amt
                if cur:
                    result["currency"] = cur
        except Exception:
            pass

    # 通貨の埋め（表示や locale からの補完）
    if result["currency"] is None:
        # URL の通貨ヒント（en-int/GB なら GBP を優先、EU は EUR など）——過剰推測は避け、最小限
        u = (page.url or "").lower()
        if "/en-gb" in u or "shiptocountry=gb" in u:
            result["currency"] = "GBP"
        elif "/en-us" in u or "shiptocountry=us" in u:
            result["currency"] = "USD"
        elif "/en-int" in u:
            # 国不明のときは EUR を安全側で
            result["currency"] = "EUR"

    # 価格が centAmount 系から来た場合の補正（_dig_price_in_dict 内でも補正しているが念のための最終整形）
    if isinstance(result.get("price"), (int, float)):
        p = float(result["price"])  # type: ignore
        # 10,000 を超える JPY 以外は小数が欲しい。JPY なら整数で自然。
        if result.get("currency") != "JPY":
            # 小数が無いのにやたら大きい場合、centAmount 誤取りの可能性
            if math.isfinite(p) and p > 10000 and abs(p % 100) < 1e-6:
                result["price"] = round(p / 100.0, 2)

    result["raw"]["text_candidates"] = text_candidates[:30]
    return result


# === 互換ラッパー（browser_use_agent が期待している API） ===================
# 以下のラッパー関数を、ご提示いただいた分析に基づき追加します。
# これにより、`browser_use_agent.py` の `from app.extractors.product_info_extractor import extract_title_price`
# が成功するようになります。
#
# （注: 以下の import はファイルの先頭で既に行われていますが、
#   このブロックを独立してコピペできるように念のため含めることがあります。
#   Pythonはモジュールのインポートをキャッシュするため実害はありません。）
from typing import Optional, Dict, Any
from playwright.async_api import Page

async def extract_title_price(
    page: Page,
    site: str = "",
    site_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    旧API互換: タイトル(=name), 価格, 通貨, URL を返す。
    browser_use_agent.py が `extract_title_price` を import するためのラッパー。
    """
    data = await extract_product_info(page, site=site, site_config=site_config)
    return {
        "title": data.get("name"),
        "price": data.get("price"),
        "currency": data.get("currency"),
        "url": data.get("url"),
        # 余裕があれば他のフィールドも渡したい場合はここで増やせる
    }

# 既存呼び出しとの互換（他所で extract() を使っているケース向け）
async def extract(page: Page, site: str = "", site_config: Optional[Dict[str, Any]] = None):
    return await extract_product_info(page, site=site, site_config=site_config)