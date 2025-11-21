# -*- coding: utf-8 -*-

"""
ai_supplier_scout.py (完全互換・統合最終版)
======================================================================
Registry: app/utils/ai_supplier_scout.py
Rev: 2025-09-07 22:00 JST

機能概要:
- Playwright(sync)による価格スカウトCLI（Windows安定運用）。
- 完全互換性・復元アップデートを統合:
  - 確定的・自己証明アーキテクチャの設定ロード（デフォルト→base→overrides→legacy追加）。
  - SSENSEセレクタを最終確定: DOM分析に基づきdata-test/id属性の優先順位を最終最適化。
  - ★検索入力/送信ロジックを最終強化し、form[role=search]スコープで確実性を向上。
  - 検索成功をURLパターンで確認し、入力セレクタを強化して最終安定化。
  - PDP遷移後の価格取得安定性を向上させるため、価格要素の可視待機ロジックを追加。
  - UI検索フローの安定性を向上: Playwrightのベストプラクティス（可視待機、汎用ロールセレクタ）を導入し、フォールバック時の信頼性を強化。
  - レガシー設定 (crawler_sites.json) をフォールバックとして読み込み。
  - PDP直指定 (--pdp-url) を完全復活。直アクセスで迅速抽出・例外時スクショ保存。
  - 検索フロー側でも例外時に確実にスクリーンショット保存（Timeout/その他）。
  - persistent context による再訪問ユーザー偽装、navigator.webdriver除去などボット回避策を維持。
  - 直テンプレートブロック検知時にUIフローへ即フォールバック。
  - --slow-mo オプションで固定遅延を適用（persistent/非persistent双方）。

--- 操作するソフト/前提 ---
- Python 3.10 以上
- 任意のエディタ/IDE（VS Code 推奨）

--- 依存ライブラリのインストール（ターミナル） ---
pip install "playwright==1.42.0"
playwright install chromium

--- 使用方法 (コマンドプロンプト or PowerShell) ---
# 1. SSENSEを人間のようにトップページから検索（ヒューマン・ファースト）
python -m app.utils.ai_supplier_scout "Givenchy bags" --sites SSENSE --headful --fx-to JPY

# 2. PDPのURLが分かっている場合（直指定・迅速抽出）
python -m app.utils.ai_supplier_scout "DUMMY" --pdp-url "https://www.ssense.com/ja-jp/men/product/givenchy/..." --headful

# 3. 動作をゆっくりにしてデバッグ（固定遅延ms）
python -m app.utils.ai_supplier_scout "Givenchy bags" --sites SSENSE --headful --slow-mo 500
======================================================================
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import random
import time
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urlparse, quote_plus
from pathlib import Path
import re as _re

from playwright.sync_api import (
    sync_playwright,
    Page,
    Browser,
    BrowserContext,
    TimeoutError as PWTimeout,
)

# === FXユーティリティ（分離モジュール） ===
# (fx_utils.py が同じ階層に存在することを想定)
try:
    from app.utils.fx_utils import (
        parse_fx_rates_str,
        get_fx_table_jpy,
    )
except ImportError:
    # フォールバック実装（モジュールが見つからない場合）
    def parse_fx_rates_str(s: str) -> dict: return {}
    def get_fx_table_jpy(**kwargs) -> tuple[dict, dict]: return {}, {"source": "fallback-dummy"}
    logging.warning("Could not import 'app.utils.fx_utils'. Using dummy FX implementation.")


# ---------------------------------------------------------
# パス/出力設定
# ---------------------------------------------------------
_p = Path(__file__).resolve()
APP_ROOT = _p.parents[2] if len(_p.parents) >= 3 else _p.parent  # 安全なルート解決
INSTANCE_ROOT = APP_ROOT / "instance"
LOG_DIR = INSTANCE_ROOT / "logs"
SS_DIR = LOG_DIR / "screenshots"
ERR_JSON = LOG_DIR / "supplier_scout_last_error.json"
USER_DATA_DIR = INSTANCE_ROOT / "pw_profile" / "supplier_scout"

CFG_LEGACY = APP_ROOT / "config" / "crawler_sites.json"
SITES_DIR = APP_ROOT / "config" / "sites"
CFG_BASE = SITES_DIR / "base.json"
CFG_OVR = SITES_DIR / "overrides.local.json"

SS_DIR.mkdir(parents=True, exist_ok=True)
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()

def ss_path(site: str, tag: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(SS_DIR / f"{ts}_{site}_{tag}.png")

def save_last_error(site: str, message: str, screenshot: Optional[str] = None, extra: Optional[dict] = None):
    payload = {"timestamp": now_iso(), "site": site, "message": message, "screenshot": screenshot}
    if extra:
        payload.update(extra)
    with open(ERR_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------
# 通貨/価格ユーティリティ
# ---------------------------------------------------------
_CURRENCY_SIGNS = {"¥": "JPY", "￥": "JPY", "$": "USD", "€": "EUR", "£": "GBP", "₩": "KRW", "₫": "VND", "A$": "AUD", "C$": "CAD"}
_CURR_WORDS = {"JPY": ["JPY", "円"], "USD": ["USD"], "EUR": ["EUR"], "GBP": ["GBP"], "AUD": ["AUD"], "CAD": ["CAD"]}
_PRICE_RE = _re.compile(r"([0-9][0-9\.,\s]*)")

def detect_currency(text: str, currency_hint: Optional[str] = None) -> str:
    if currency_hint:
        return currency_hint.upper()
    for sign, code in _CURRENCY_SIGNS.items():
        if sign in text:
            return code
    up = text.upper()
    for code, words in _CURR_WORDS.items():
        if any(w in up for w in words):
            return code
    return "UNKNOWN"

def to_number(price_text: str) -> Optional[int]:
    t = price_text.replace("\u00a0", " ")
    m = _PRICE_RE.search(t)
    if not m:
        return None
    digits = _re.sub(r"[^\d]", "", m.group(1))
    return int(digits) if digits else None

def convert_price(price: int, currency: str, fx_to: Optional[str], fx_table: Dict[str, float]) -> Optional[float]:
    if not fx_to:
        return None
    fx_to = fx_to.upper()
    currency = (currency or "").upper()
    if fx_to == "JPY":
        if currency == "JPY":
            return float(price)
        rate = fx_table.get(currency)
        return price * rate if rate else None
    return None

# ---------------------------------------------------------
# サイト設定モデル
# ---------------------------------------------------------
@dataclass
class SiteSelectors:
    search_open: List[str] = field(default_factory=list)
    search_input: List[str] = field(default_factory=list)
    search_submit: List[str] = field(default_factory=list)
    results_item: Optional[str] = None
    first_product_link: Optional[str] = None
    pdp_title: List[str] = field(default_factory=list)
    pdp_price: List[str] = field(default_factory=list)

@dataclass
class SiteConfig:
    name: str
    home_url: str
    domains: List[str] = field(default_factory=list)
    search_mode: str = "human"
    search_template: Optional[str] = None
    wait_until: str = "domcontentloaded"
    timeout_sec: int = 25
    currency_hint: Optional[str] = None
    selectors: SiteSelectors = field(default_factory=SiteSelectors)
    force_ui_search: bool = False
    notes: Optional[str] = None

# ---------------------------------------------------------
# デフォルトサイト（堅牢なフォールバックセレクタ）
# ---------------------------------------------------------
def default_sites() -> List[SiteConfig]:
    return [
        SiteConfig(
            name="SSENSE",
            home_url="https://www.ssense.com/ja-jp",
            domains=["ssense.com"],
            search_template="https://www.ssense.com/ja-jp/search?q={q}",
            selectors=SiteSelectors(
                search_open=[
                    "i.fa-ssense-magnifier",
                    "a[data-test='mobileNavigationSearchLink']",
                    "a.mobile-header-search",
                    "button[aria-label='Open Search']",
                ],
                search_input=[
                    "form[role='search'] input#search-form-input",
                    "#search-form-input",
                    "form[role='search'] input[aria-label='Search']",
                    "input[aria-label='Search']",
                    "input[data-testid='search-input']",
                    "input[type='search']",
                    "input[name='q']",
                ],
                search_submit=[
                    "#searchSubmitIcon",
                    "button[type='submit']",
                ],
                results_item="a[href*='/product/']",
                first_product_link="a[href*='/product/']",
                pdp_title=[
                    "#pdpProductNameText",
                    "[data-test='pdpProductNameText']",
                    "h1",
                    "h2#pdpProductNameText",
                    ".pdp-product-title__name",
                ],
                pdp_price=[
                    "[data-test='pdpRegularPriceText']",
                    "span.product-price-mobile__price",
                    ".product-price__sale",
                ],
            ),
        ),
        SiteConfig(
            name="BUYMA",
            home_url="https://www.buyma.com/",
            domains=["buyma.com"],
            search_template="https://www.buyma.com/r/-/search/?q={q}",
            selectors=SiteSelectors(
                search_input=["#search_txt", "input.fab-search-txtarea", "input#srchTxt"],
                search_submit=["form#search_form", "button#srchBtn"],
                results_item="a[href*='/item/']",
                first_product_link="a[href*='/item/']",
                pdp_title=["h1[itemprop='name']", "h1.product_title", "h1"],
                pdp_price=["span.Price_Txt", "#price", ".product_price .Price_Txt", "span[itemprop='price']"],
            ),
        ),
        SiteConfig(
            name="FARFETCH",
            home_url="https://www.farfetch.com/",
            domains=["farfetch.com"],
            search_template="https://www.farfetch.com/shopping/men/items.aspx?q={q}",
            selectors=SiteSelectors(
                results_item="a[data-testid='productCard-link'], a[href*='/shopping/']",
                first_product_link="a[data-testid='productCard-link'], a[href*='/shopping/']",
                pdp_title=["h1[data-tstid='product-name']", "h1", "span[itemprop='name']"],
                pdp_price=[
                    "[data-tstid='priceInfo-original']",
                    "[data-tstid='priceInfo-onsale']",
                    "p[data-tstid='priceInfo']",
                    "span[data-tstid='current-price']",
                ],
            ),
        ),
        SiteConfig(
            name="MATCHES",
            home_url="https://www.matchesfashion.com/",
            domains=["matchesfashion.com"],
            search_template="https://www.matchesfashion.com/intl/search?text={q}",
            selectors=SiteSelectors(
                results_item="a[href*='/products/']",
                first_product_link="a[href*='/products/']",
                pdp_title=["h1", "h1[data-test='pdp-title']", "div[data-test='pdp-title']"],
                pdp_price=[
                    "span[data-test='pdp-price']",
                    "div.prices span.price",
                    "span.now",
                    "span.was",
                ],
            ),
        ),
        SiteConfig(
            name="GENERIC_PDP",
            home_url="",
            domains=[],
            selectors=SiteSelectors(
                pdp_title=["h1", "title"],
                pdp_price=[
                    "meta[itemprop='price']",
                    "span[itemprop='price']",
                    "span:has-text('¥')",
                ],
            ),
            notes="--pdp-url",
        ),
    ]

# ---------------------------------------------------------
# 設定レイヤー読み込み（base/overrides優先 + legacyフォールバック）
# ---------------------------------------------------------
def _update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = _update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def _load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"Could not load or parse JSON from {path}: {e}")
        return {}

def _dict_to_siteconfig(d: dict) -> SiteConfig:
    # SiteConfig/Selectorsを安全に再構築（旧search/pdp階層にも対応）
    sels = d.get("selectors", {})
    if not isinstance(sels, dict):
        sels = {}
    # 旧階層→フラットへ吸収
    sel_flat = {
        "search_open": (sels.get("search", {}) or {}).get("open") or sels.get("search_open", []),
        "search_input": (sels.get("search", {}) or {}).get("input") or sels.get("search_input", []),
        "search_submit": (sels.get("search", {}) or {}).get("submit") or sels.get("search_submit", []),
        "results_item": (sels.get("search", {}) or {}).get("results_item") or sels.get("results_item"),
        "first_product_link": (sels.get("search", {}) or {}).get("first_product_link") or sels.get("first_product_link"),
        "pdp_title": (sels.get("pdp", {}) or {}).get("title") or sels.get("pdp_title", []),
        "pdp_price": (sels.get("pdp", {}) or {}).get("price") or sels.get("pdp_price", []),
    }
    sc = SiteConfig(
        name=d.get("name") or "UNKNOWN",
        home_url=d.get("home_url") or "",
        domains=d.get("domains") or [],
        search_mode=d.get("search_mode", "human"),
        search_template=d.get("search_template"),
        wait_until=d.get("wait_until", "domcontentloaded"),
        timeout_sec=int(d.get("timeout_sec", 25)),
        currency_hint=d.get("currency_hint"),
        selectors=SiteSelectors(**sel_flat),
        force_ui_search=d.get("force_ui_search", False),
        notes=d.get("notes"),
    )
    return sc

def load_config_sites() -> List[SiteConfig]:
    log = logging.getLogger("ConfigLoader")
    loaded_files = []
    # デフォルトをdict化
    sites_map: Dict[str, dict] = {s.name.upper(): asdict(s) for s in default_sites()}

    # baseを適用
    base_data = _load_json_if_exists(CFG_BASE)
    if base_data:
        loaded_files.append(str(CFG_BASE.relative_to(APP_ROOT)))
        for site_conf in base_data.get("sites", []):
            name = (site_conf.get("name") or "").upper()
            if name in sites_map:
                sites_map[name] = _update_dict(sites_map[name], site_conf)

    # overridesを確定的に上書き
    overrides_data = _load_json_if_exists(CFG_OVR)
    if overrides_data:
        loaded_files.append(str(CFG_OVR.relative_to(APP_ROOT)))
        for site_conf in overrides_data.get("sites", []):
            name = (site_conf.get("name") or "").upper()
            if name in sites_map:
                sites_map[name] = _update_dict(sites_map[name], site_conf)

    # legacyは未知サイトのみ追加（フォールバック）
    legacy_data = _load_json_if_exists(CFG_LEGACY)
    if legacy_data:
        loaded_files.append(str(CFG_LEGACY.relative_to(APP_ROOT)))
        for site_conf in legacy_data.get("sites", []):
            name = (site_conf.get("name") or "").upper()
            if name and name not in sites_map:
                sites_map[name] = site_conf
                log.info(f"Loaded '{name}' from legacy config as fallback.")

    log.info(f"Config loaded from: {', '.join(loaded_files) if loaded_files else 'defaults only'}")
    # 追加の自己証明ログ（存在する実パス）
    log.info(f"CFG_BASE:   {CFG_BASE}")
    log.info(f"CFG_OVR:    {CFG_OVR}")
    log.info(f"CFG_LEGACY: {CFG_LEGACY}")

    # dict→SiteConfig
    return [_dict_to_siteconfig(d) for d in sites_map.values()]

# ---------------------------------------------------------
# スカウト本体
# ---------------------------------------------------------
class SupplierScout:
    def __init__(
        self,
        headful=False,
        log_level="INFO",
        sites: Optional[List[str]] = None,
        attempts=2,
        manual_verify=False,
        manual_wait_sec=240,
        fx_to: Optional[str] = None,
        fx_rates: Optional[str] = None,
        fx_auto: bool = True,
        fx_ttl_hours: int = 12,
        slow_mo: Optional[int] = None,
    ):
        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s: %(name)s: %(message)s",
        )
        self.log = logging.getLogger("SupplierScout")
        self.headful = headful
        self.attempts = attempts
        self.manual = manual_verify
        self.manual_wait_sec = manual_wait_sec
        self.slow_mo = slow_mo

        self.fx_to = fx_to.upper() if fx_to else None
        manual_table = parse_fx_rates_str(fx_rates or "")
        auto_table, fx_meta = get_fx_table_jpy(
            auto=fx_auto, ttl_hours=fx_ttl_hours, manual_table=manual_table
        )
        self.fx_table = auto_table if auto_table else manual_table
        self.fx_meta = fx_meta
        self.log.info(
            f"FX init: to={self.fx_to}, source={self.fx_meta.get('source')}, asof={self.fx_meta.get('asof')}"
        )

        all_sites = load_config_sites()
        if sites:
            allow = set(s.upper() for s in sites)
            self.sites = [s for s in all_sites if s.name.upper() in allow]
        else:
            self.sites = all_sites

        # 自己証明ログ
        self.log.info(
            f"SupplierScout ready: headful={self.headful}, slow_mo={self.slow_mo or 'dynamic'}, sites={[s.name for s in self.sites]}"
        )
        for s in self.sites:
            self.log.info(f"Site config: {s.name} force_ui_search={s.force_ui_search} template={'yes' if s.search_template else 'no'}")

    # 待機（slow_mo優先、なければランダム）
    def _human_wait_random(self, min_ms=400, max_ms=1200):
        if self.slow_mo:
            time.sleep(self.slow_mo / 1000)
        else:
            time.sleep(random.randint(min_ms, max_ms) / 1000)

    def _accept_popups(self, page: Page):
        selectors = [
            "button:has-text('Accept All')",
            "button:has-text('すべて承認')",
            "button:has-text('Agree')",
            "button:has-text('同意する')",
            "button:has-text('I agree')",
            "button:has-text('同意')",
            "#onetrust-accept-btn-handler",
            "[aria-label='Close']",
            "[aria-label='閉じる']",
        ]
        for sel in selectors:
            try:
                page.locator(sel).first.click(timeout=1500)
                self.log.info(f"Clicked popup: {sel}")
                self._human_wait_random(700, 1500)
                return
            except Exception:
                pass

    # ---- PDP直指定（完全復活 + 例外時スクショ）
    def _run_pdp_direct_core(self, browser: Browser, site: SiteConfig, pdp_url: str) -> Optional[dict]:
        page = None
        try:
            page = browser.new_page()
            page.set_default_timeout(site.timeout_sec * 1000)
            self.log.info(f"[{site.name}] Navigating directly to PDP: {pdp_url}")
            page.goto(pdp_url, wait_until=site.wait_until)
            self._accept_popups(page)

            html = page.content() or ""
            if any(t in html for t in ["長押し", "Press and hold", "I'm not a robot"]):
                if self.manual:
                    self._human_wait(site, page)
                else:
                    raise RuntimeError("human-verification page detected")

            # PDP価格要素の可視待機
            self._wait_visible_any(page, site.selectors.pdp_price, timeout_ms=site.timeout_sec * 1000)

            title = (self._first_text(page, site.selectors.pdp_title) or "").strip()
            price_raw = self._first_text(page, site.selectors.pdp_price) or page.inner_text("body")
            currency = detect_currency(price_raw or "", site.currency_hint)
            price = to_number(price_raw or "")
            if not price:
                raise RuntimeError("could not parse price from PDP body")
            price_conv = convert_price(price, currency, self.fx_to, self.fx_table)

            self.log.info(f"[{site.name}] PDP OK: {currency} {price} — {title[:60]}")
            return {
                "site": site.name,
                "url": page.url,
                "title": title,
                "price": price,
                "currency": currency,
                "price_converted": price_conv,
                "fx_to": self.fx_to,
            }
        except Exception as e:
            tag = "pdp_timeout" if isinstance(e, PWTimeout) else "pdp_error"
            p = ss_path(site.name, tag)
            try:
                if page:
                    page.screenshot(path=p)
            except Exception:
                p = None
            save_last_error(site.name, f"[{site.name}] PDP-direct failed: {e}", p, {"pdp_url": pdp_url})
            self.log.warning(f"[{site.name}] PDP-direct failed: {e.__class__.__name__} (screenshot: {p})")
            return None
        finally:
            if page:
                page.close()

    # ---- 検索→PDP抽出の中核（直テンプレブロック検知→UIフォールバック、例外時スクショ→再送出）
    def _run_site_core(self, context: BrowserContext, site: SiteConfig, query: str) -> Optional[dict]:
        page = context.new_page()
        page.set_default_timeout(site.timeout_sec * 1000)
        try:
            if site.force_ui_search or not site.search_template:
                self.log.info(f"[{site.name}] Using Human-First UI Interaction flow.")
                page.goto(site.home_url, wait_until=site.wait_until)
                self._accept_popups(page)
                self._human_wait_random(1200, 3000)
                try:
                    page.mouse.move(random.randint(200, 800), random.randint(200, 600))
                except Exception:
                    pass
                self._search_ui(page, site, query)
            else:
                self.log.info(f"[{site.name}] Using direct search template.")
                url = site.search_template.format(q=quote_plus(query))
                page.goto(url, wait_until=site.wait_until)
                self._accept_popups(page)
                html = page.content() or ""
                # ブロック/404検知 → UIへフォールバック
                blocked = (
                    "ページが見つかりません" in html or
                    "大変申し訳ありません" in html or
                    "not found" in (html or "").lower()[:300]
                )
                if blocked:
                    self.log.warning(f"[{site.name}] direct template flagged; falling back to UI flow")
                    page.goto(site.home_url, wait_until=site.wait_until)
                    self._accept_popups(page)
                    self._human_wait_random(1200, 3000)
                    self._search_ui(page, site, query)
                else:
                    if any(t in html for t in ["長押し", "Press and hold", "I'm not a robot"]):
                        if self.manual:
                            self._human_wait(site, page)
                        else:
                            raise RuntimeError("human-verification page detected")

            link_sel = site.selectors.first_product_link or site.selectors.results_item
            if not link_sel:
                raise RuntimeError("no result link selector")

            self._wait_visible_any(page, [link_sel])
            first = page.locator(link_sel).first
            self._human_wait_random()
            first.hover(timeout=2000)
            self._human_wait_random(400, 900)
            first.click()
            page.wait_for_load_state(site.wait_until)
            self._accept_popups(page)

            # 【安定化改善】PDP遷移後、価格要素が描画されるまで待機
            self.log.info(f"[{site.name}] Waiting for PDP price element to be visible...")
            self._wait_visible_any(page, site.selectors.pdp_price)

            try:
                page.mouse.wheel(0, random.randint(200, 600))
                self._human_wait_random(400, 900)
            except Exception:
                pass
            try:
                page.screenshot(path=ss_path(site.name, "pdp"))
            except Exception:
                pass

            title = (self._first_text(page, site.selectors.pdp_title) or "").strip()
            price_raw = self._first_text(page, site.selectors.pdp_price) or page.inner_text("body")
            currency = detect_currency(price_raw or "", site.currency_hint)
            price = to_number(price_raw or "")
            if not price:
                raise RuntimeError("could not parse price")

            price_conv = convert_price(price, currency, self.fx_to, self.fx_table)
            self.log.info(
                f"[{site.name}] OK: {currency} {price} ({'~'+str(int(price_conv))+' '+self.fx_to if price_conv else 'no-fx'}) — {title[:60]}"
            )
            return {
                "site": site.name,
                "url": page.url,
                "title": title,
                "price": price,
                "currency": currency,
                "price_converted": price_conv,
                "fx_to": self.fx_to,
            }
        except Exception:
            tag = "timeout" if isinstance(sys.exc_info()[1], PWTimeout) else "error"
            p = ss_path(site.name, f"{tag}_core")
            try:
                page.screenshot(path=p)
            except Exception:
                p = None
            raise
        finally:
            page.close()

    def run_site_with_context(self, context: BrowserContext, site: SiteConfig, query: str) -> Optional[dict]:
        self.log.info(f"--- Running site: {site.name} (force_ui_search: {site.force_ui_search}) ---")
        for attempt in range(1, self.attempts + 1):
            try:
                self.log.info(f"[{site.name}] attempt {attempt} start")
                return self._run_site_core(context, site, query)
            except PWTimeout as e:
                p = ss_path(site.name, f"timeout_attempt_{attempt}")
                self._try_screenshot_context(context, p)
                save_last_error(site.name, f"[{site.name}] TIMEOUT on attempt {attempt}: {e}", p)
                self.log.warning(f"[{site.name}] TIMEOUT on attempt {attempt}: {e.__class__.__name__} (screenshot: {p})")
            except Exception as e:
                p = ss_path(site.name, f"error_attempt_{attempt}")
                self._try_screenshot_context(context, p)
                save_last_error(site.name, f"[{site.name}] ERROR on attempt {attempt}: {e}", p)
                self.log.warning(f"[{site.name}] ERROR on attempt {attempt}: {e.__class__.__name__} (screenshot: {p})", exc_info=False)
        self.log.error(f"[{site.name}] giving up after {self.attempts} attempts.")
        return None

    def _try_screenshot_context(self, context: BrowserContext, path: str) -> bool:
        try:
            pages = context.pages
            if pages:
                pages[-1].screenshot(path=path)
                return True
        except Exception:
            pass
        return False

    def _search_ui(self, page: Page, site: SiteConfig, query: str):
        # 1) 検索オープン（虫眼鏡 or Searchボタン）
        opened = False
        for sel in site.selectors.search_open or []:
            try:
                loc = page.locator(sel).first
                loc.wait_for(state="visible", timeout=4000)
                loc.click()
                self._human_wait_random(400, 900)
                opened = True
                break
            except Exception:
                continue
        if not opened:
            # 汎用フォールバック（役割検索）
            try:
                page.get_by_role("button", name=_re.compile("Search|検索|さがす", _re.I)).click(timeout=4000)
                self._human_wait_random(400, 900)
                opened = True
            except Exception:
                pass

        # 2) 入力（フォールバック順で探索）
        input_found = False
        # ★【安定化改善】設定ファイルから候補を取得
        candidates = site.selectors.search_input
        for css in candidates:
            try:
                inp = page.locator(css).first
                inp.wait_for(state="visible", timeout=5000)
                inp.click()
                self._human_wait_random(200, 500)
                inp.fill(query, timeout=5000)
                input_found = True
                break
            except Exception:
                continue

        if not input_found:
            raise RuntimeError(f"[{site.name}] no usable search input")

        # 3) 送信（Enter優先→ボタン明示→定義済みsubmitの順）
        submitted = False
        try:
            page.keyboard.press("Enter")
            submitted = True
        except Exception:
            pass

        if not submitted:
            try:
                page.locator("#searchSubmitIcon").first.click(timeout=3000)
                submitted = True
            except Exception:
                pass

        if not submitted:
            for sel in site.selectors.search_submit or []:
                try:
                    page.locator(sel).first.click(timeout=2000)
                    submitted = True
                    break
                except Exception:
                    continue

        # 検索結果URLへの遷移を待機（成功判定）
        page.wait_for_url(r"**/search?q=*", timeout=30000)
        page.wait_for_load_state(site.wait_until)


    def _first_text(self, page, selectors: List[str]) -> Optional[str]:
        for sel in selectors:
            try:
                loc = page.locator(sel).first
                if loc.count() > 0:
                    t = loc.inner_text(timeout=5000).strip()
                    if t:
                        return t
            except Exception:
                continue
        return None

    def _wait_visible_any(self, page, selectors: List[str], timeout_ms: int = 25000):
        last_err = None
        for sel in selectors:
            try:
                page.locator(sel).first.wait_for(state="visible", timeout=timeout_ms)
                return
            except Exception as e:
                last_err = e
                continue
        raise PWTimeout(f"wait any visible failed for selectors: {selectors} (last error: {last_err})")

    def _human_wait(self, site: SiteConfig, page):
        self.log.warning(f"[HUMAN-CHECK] 検知: 手動解除を待機（~{self.manual_wait_sec}秒）")
        remain = self.manual_wait_sec
        while remain > 0:
            self.log.warning(f"[HUMAN-CHECK] 継続中… 残り ~{remain} 秒")
            page.wait_for_timeout(5000)
            remain -= 5
            html = page.content() or ""
            if all(t not in html for t in ["長押し", "Press and hold", "I'm not a robot"]):
                return
        raise RuntimeError("human-verification not cleared (timeout)")

    def run_once(self, product: str, pdp_url: Optional[str] = None) -> dict:
        results: List[dict] = []
        with sync_playwright() as pw:
            if pdp_url:
                self.log.info(f"PDP direct URL mode activated for: {pdp_url}")
                parsed_url = urlparse(pdp_url)
                target_site = next((s for s in self.sites if any(d in parsed_url.netloc for d in s.domains)), None)
                if not target_site:
                    target_site = next((s for s in self.sites if s.name.upper() == "GENERIC_PDP"), self.sites[0])
                    self.log.warning(f"Domain not matched. Using fallback site config: {target_site.name}")
                browser = pw.chromium.launch(headless=not self.headful, slow_mo=self.slow_mo or 0)
                try:
                    r = self._run_pdp_direct_core(browser, target_site, pdp_url)
                    if r:
                        results.append(r)
                finally:
                    browser.close()
            else:
                self.log.info(f"Using persistent profile: {USER_DATA_DIR}")
                context = pw.chromium.launch_persistent_context(
                    str(USER_DATA_DIR),
                    headless=not self.headful,
                    slow_mo=self.slow_mo or 0,
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
                    viewport={'width': 1920, 'height': 1080},
                    screen={'width': 1920, 'height': 1080},
                    locale='ja-JP',
                    timezone_id='Asia/Tokyo',
                    args=['--start-maximized', '--disable-blink-features=AutomationControlled'],
                    extra_http_headers={
                        "Accept-Language": "ja-JP,ja;q=0.9,en-US;q=0.8,en;q=0.7",
                        "Upgrade-Insecure-Requests": "1",
                        "DNT": "1",
                    },
                )
                context.add_init_script("Object.defineProperty(navigator, 'webdriver', { get: () => undefined });")
                try:
                    for site in self.sites:
                        if site.name.upper() == "GENERIC_PDP":
                            continue
                        r = self.run_site_with_context(context, site, product)
                        if r:
                            results.append(r)
                finally:
                    context.close()

        if not results:
            return {"ok": False, "message": "No valid supplier found."}

        def best_key(x):
            price_converted = x.get("price_converted")
            price = x.get("price", float("inf"))

            # 換算価格が存在すればそれを優先、なければ元の価格で比較
            primary_key = float(price_converted) if price_converted is not None else float("inf")
            secondary_key = float(price)

            return (primary_key, secondary_key)

        sorted_results = sorted(results, key=best_key)
        best_item = sorted_results[0]

        return {"ok": True, "fx": self.fx_meta, "best": {**best_item, "timestamp": now_iso()}, "results": sorted_results}


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description=__doc__)
    p.add_argument("product", help="検索用の製品キーワード")
    p.add_argument("--sites", nargs="+", help="対象サイト（例: FARFETCH MATCHES SSENSE BUYMA）")
    p.add_argument("--headful", action="store_true", help="ヘッドあり起動（動作確認用）")
    p.add_argument("--log-level", default="INFO", help="DEBUG/INFO/WARN/ERROR")
    p.add_argument("--attempts", type=int, default=2, help="各サイトの試行回数")
    p.add_argument("--manual-verify", action="store_true", help="人間検証が出たら手動待機")
    p.add_argument("--manual-wait-sec", type=int, default=240, help="手動待機の最大秒数")
    p.add_argument("--pdp-url", help="PDP直指定URL（手動突破後のURL）")
    p.add_argument("--fx-to", help="換算先通貨（JPY）")
    p.add_argument("--fx-rates", help="手動レート 'USD=155,EUR=170'")
    p.add_argument("--fx-auto", dest="fx_auto", action="store_true", default=True, help="為替レートを自動取得 ※既定: ON")
    p.add_argument("--no-fx-auto", dest="fx_auto", action="store_false", help="為替レートの自動取得を無効化")
    p.add_argument("--fx-ttl-hours", type=int, default=12, help="為替キャッシュのTTL(時間)")
    p.add_argument("--no-startup-fx-refresh", action="store_true", help="起動時の為替自動更新を無効化")
    p.add_argument("--slow-mo", type=int, help="操作間の固定遅延(ms)。デバッグ用。")
    args = p.parse_args()

    if not args.no_startup_fx_refresh and os.getenv("FX_AUTO_STARTUP", "1") == "1":
        try:
            _, meta = get_fx_table_jpy(auto=True, ttl_hours=24, manual_table={})
            print(f"[FX] Auto-refresh: asof={meta.get('asof')}, source={meta.get('source')}")
        except Exception:
            print("[FX] Auto-refresh skipped due to error")

    scout = SupplierScout(
        headful=args.headful,
        log_level=args.log_level,
        sites=args.sites,
        attempts=args.attempts,
        manual_verify=args.manual_verify,
        manual_wait_sec=args.manual_wait_sec,
        fx_to=args.fx_to,
        fx_rates=args.fx_rates,
        fx_auto=args.fx_auto,
        fx_ttl_hours=args.fx_ttl_hours,
        slow_mo=args.slow_mo,
    )
    result = scout.run_once(args.product, pdp_url=args.pdp_url)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1

if __name__ == "__main__":
    # モジュールとして実行された場合に備え、sys.exitを直接呼ばない
    # 例: `python -m app.utils.ai_supplier_scout ...`
    if len(sys.argv) > 1:
        sys.exit(main())
