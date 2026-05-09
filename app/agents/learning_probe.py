# learning_probe.py — 偵察モードで「価格とリンクの在処」を学習し、learned_selectors.json に保存
from __future__ import annotations

import json
import logging
import re
from typing import Any
from urllib.parse import urljoin

from playwright.async_api import Page

log = logging.getLogger(__name__)

JSONLD_SEL = "script[type='application/ld+json']"

PRICE_PAT = re.compile(
    r"(?P<curr>[€$£¥]|AUD|CAD|CHF|CNY|EUR|GBP|HKD|JPY|KRW|TWD|USD)\s*[\d,.]+|[\d,.]+\s*(?P<curr2>[€$£¥])"
)
NEAR_PRICE_HINTS = re.compile(r"(price|税込|tax|含税|vat|prix|precio|prezzo)", re.I)


def _generalize_css(sel: str) -> str:
    # 揮発的な修飾子を削り、安定属性を優先
    sel = re.sub(r"--[a-z0-9_-]+", "", sel)  # BEM修飾子
    sel = re.sub(r"\.[a-z0-9_-]*(?:\d|_[a-z0-9]{4,})", "", sel)  # ハッシュ/連番っぽいclass
    sel = re.sub(r":nth-child\(\d+\)", "", sel)
    return sel


async def _extract_jsonld_price(page: Page) -> dict[str, Any] | None:
    nodes = page.locator(JSONLD_SEL)
    try:
        count = await nodes.count()
    except Exception:
        count = 0
    for i in range(count):
        try:
            raw = await nodes.nth(i).inner_text()
            data = json.loads(raw)
            # Product or Offer系を探す
            candidates = []
            if isinstance(data, dict):
                candidates.append(data)
            elif isinstance(data, list):
                candidates.extend([x for x in data if isinstance(x, dict)])
            for d in candidates:
                if d.get("@type") in ("Product", "AggregateOffer", "Offer") or "offers" in d:
                    offers = d.get("offers", d if "price" in d else {})
                    if isinstance(offers, list):
                        offers = next(
                            (o for o in offers if isinstance(o, dict) and ("price" in o or "priceCurrency" in o)), {}
                        )
                    price = offers.get("price")
                    cur = offers.get("priceCurrency")
                    if price:
                        return {"price": str(price), "currency": cur or "", "source": "jsonld"}
        except Exception:
            continue
    return None


async def _extract_text_price(page: Page) -> dict[str, Any] | None:
    # 可視テキストから価格らしき塊を拾い、意味語に近いノードを優先
    try:
        entries = await page.evaluate("""
        () => {
          const out = [];
          const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT);
          while (walker.nextNode()) {
            const el = walker.currentNode;
            const style = window.getComputedStyle(el);
            if (!el || style.display === 'none' || style.visibility === 'hidden') continue;
            const txt = (el.textContent || '').trim();
            if (txt && txt.length < 64) out.push([txt, el.tagName.toLowerCase(), el.className || '', el.getAttribute('itemprop') || '', el.getAttribute('data-testid') || '']);
          }
          return out;
        }
        """)
    except Exception:
        entries = []
    scored = []
    for text, tag, cls, itemprop, dtid in entries:
        if PRICE_PAT.search(text):
            score = 1
            if itemprop and "price" in itemprop.lower():
                score += 3
            if dtid and "price" in dtid.lower():
                score += 2
            if NEAR_PRICE_HINTS.search(text):
                score += 1
            scored.append((score, text, tag, cls))
    if not scored:
        return None
    scored.sort(reverse=True)
    best = scored[0]
    return {"price_text": best[1], "source": "text"}


async def learn_on_pdp(page: Page) -> dict[str, Any]:
    """PDPで価格抽出の“勝ち筋”を学ぶ"""
    # 1) JSON-LD 最優先
    data = await _extract_jsonld_price(page)
    price_ok = bool(data and data.get("price"))
    # 2) テキスト補助
    if not price_ok:
        text = await _extract_text_price(page)
        if text:
            data = {**(data or {}), **text}

    # 価格ノードのCSS候補（汎化）を列挙
    css_candidates = await page.evaluate("""
      () => {
        const qs = [
          "[itemprop='price']",
          "[data-testid*='price' i]",
          "[class*='price' i]",
          "meta[itemprop='price']"
        ];
        const nodes = Array.from(document.querySelectorAll(qs.join(',')));
        const out = nodes.slice(0, 10).map(n => {
          try {
            let sel = '';
            if (n.id) sel = '#' + n.id;
            else if (n.classList && n.classList.length) sel = '.' + Array.from(n.classList).slice(0,2).join('.');
            else sel = n.tagName.toLowerCase();
            return sel || '';
          } catch(e){ return ''; }
        }).filter(Boolean);
        return Array.from(new Set(out));
      }
    """)
    css_general = [_generalize_css(s) for s in (css_candidates or [])]
    learned = {
        "price_data": data or {},
        "price_selectors": [s for s in css_general if s][:5]
        + ["[itemprop='price']", "[data-testid*='price' i]", ".price"],
        "title_selectors": ["h1", "[data-testid='product-name']", ".c-product-title"],
    }
    return learned


async def learn_plp_links(page: Page, base_url: str) -> dict[str, Any]:
    # PLPからPDPリンクを得るセレクタを学ぶ
    hrefs = await page.evaluate("""
      () => {
        const sels = [
          "[data-qa='product-tile'] a[href]",
          "a[data-qa='product-card-link']",
          "a[data-testid*='product' i][href]",
          "a[href*='/products/']",
          "a[href*='/p/']",
          ".product-card a[href]"
        ];
        const nodes = Array.from(document.querySelectorAll(sels.join(',')));
        return Array.from(new Set(nodes.map(n => n.href || n.getAttribute('href') || ''))).filter(Boolean);
      }
    """)
    return {
        "pdp_link_selectors": [
            "[data-qa='product-tile'] a[href]",
            "a[data-qa='product-card-link']",
            "a[data-testid*='product' i][href]",
            "a[href*='/products/']",
            "a[href*='/p/']",
            ".product-card a[href]",
        ],
        "sample_pdp_urls": [urljoin(base_url, h) for h in hrefs][:10],
    }


def merge_learned(original: dict[str, Any], learned: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(original)) if original else {}

    def _merge_list(dst_key: str, values: list[str]):
        ex = list(dict.fromkeys((out.get(dst_key) or []) + (values or [])))
        out[dst_key] = ex

    if learned.get("price_selectors"):
        _merge_list("price_selectors", learned["price_selectors"])
    if learned.get("title_selectors"):
        _merge_list("title_selectors", learned["title_selectors"])
    if learned.get("pdp_link_selectors"):
        _merge_list("pdp_link_selectors", learned["pdp_link_selectors"])
    # 参考値としてサンプルURLも保持
    if learned.get("sample_pdp_urls"):
        out["sample_pdp_urls"] = learned["sample_pdp_urls"]
    return out
