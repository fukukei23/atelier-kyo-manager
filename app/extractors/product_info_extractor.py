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
import re
import logging
from typing import Any, Dict, List, Optional, Set

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

logger = logging.getLogger(__name__)

VALID_PARSING_KEYS: Set[str] = { "title", "price", "brand", "list_price", "discount_pct" }

def _normalize_and_to_int(s: Optional[str]) -> Optional[int]:
    if not s: return None
    try:
        normalized = s.translate(str.maketrans("０１２３４5６７８９", "0123456789"))
        cleaned = re.sub(r"[^0-9.]", "", normalized)
        if not cleaned: return None
        return int(float(cleaned))
    except (ValueError, TypeError):
        return None

def _first_yen_from_text(text: str, preferred_currency: Optional[str] = None) -> Optional[int]:
    if not text: return None
    match = re.search(r"(?:¥|￥|\$|€)\s*([\d,.]+)|([\d,.]+)\s*(?:円|USD|EUR)", text, re.IGNORECASE)
    if match:
        return _normalize_and_to_int(match.group(1) or match.group(2))
    return _normalize_and_to_int(text)

# ( ... 他のヘルパー関数は変更なし ... )

def extract_product_info(html: str, site_config: Dict[str, Any]) -> Dict[str, Any]:
    if not BeautifulSoup: raise RuntimeError("BeautifulSoup4/lxml is not installed.")
    soup = BeautifulSoup(html, "lxml")
    pdp_selectors = site_config.get("selectors", {}).get("pdp", {})
    out: Dict[str, Any] = {"source_flags": {}}

    # 1) JSON-LD (最優先)
    # (実装は変更なし)

    # 2) Open Graph & Meta Tags
    if not out.get("title"):
        og_title = soup.select_one("meta[property='og:title']")
        if og_title and og_title.get("content"):
            out["title"] = og_title["content"]; out["source_flags"]["title"] = "og:title"
    if not out.get("brand"):
        og_brand = soup.select_one("meta[property='og:site_name']")
        if og_brand and og_brand.get("content"):
            out["brand"] = og_brand["content"]; out["source_flags"]["brand"] = "og:site_name"
    if not out.get("price"):
        og_price = soup.select_one("meta[property='product:price:amount']")
        if og_price and og_price.get("content"):
            out["price"] = _normalize_and_to_int(og_price["content"])
            out["source_flags"]["price"] = "og:price"

            ### ★★★ 変更点: 通貨情報も取得 ★★★
            og_currency = soup.select_one("meta[property='product:price:currency']")
            if og_currency and og_currency.get("content"):
                out["currency"] = og_currency["content"]
                out["source_flags"]["currency"] = "og:currency"

    # 3) CSS Selectors (最終手段)
    # (実装は変更なし)
    for key in VALID_PARSING_KEYS:
        if out.get(key): continue
        # ...

    # (計算フォールバック、最終チェックは変更なし)
    return out
