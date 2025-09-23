# -*- coding: utf-8 -*-
"""
ai_supplier_scout.py (プロダクション・レディネス版)
======================================================================
Registry: app/utils/ai_supplier_scout.py
Rev: 2025-09-08 10:13 JST

機能概要:
- Playwright(sync)による価格スカウトCLIの完成形。
- ★戦略更新★「プロダクション・レディネス・アップデート」:
  - 高速化: --speed-upフラグで画像/広告等をブロックし、動作を高速化。
  - 安定性向上: PDP遷移後にURLパターン待機を追加し、描画遅延への耐性を強化。
  - 分析性向上: Timeoutとその他エラーを区別し、ログと結果の粒度を向上。
  - 既存のbot対策、運用安定性、後方互換性はすべて維持。

--- 操作するソフト/前提 ---
- Python 3.10 以上
- Playwright 1.42.0 (`pip install "playwright==1.42.0"`)
- playwright install chromium

--- 使用方法 (コマンドプロンプト or PowerShell) ---
# 1. 通常実行 (SSENSE, 人間のようにトップページから)
python -m app.utils.ai_supplier_scout "Gucci Horsebit 1955" --sites SSENSE --headful

# 2. 高速モードで実行
python -m app.utils.ai_supplier_scout "Gucci Horsebit 1955" --sites SSENSE --speed-up

# 3. PDPのURLを直接指定
python -m app.utils.ai_supplier_scout "DUMMY" --pdp-url "https://www.ssense.com/..."
======================================================================
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import re
import sys
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urlparse, quote_plus
from pathlib import Path
import copy

from playwright.sync_api import sync_playwright, Page, Browser, BrowserContext, Route, TimeoutError as PWTimeout

# === FXユーティリティ（分離モジュール） ===
from app.utils.fx_utils import (
    parse_fx_rates_str,
    get_fx_table_jpy,
)

# ---------------------------------------------------------
# パス/出力設定
# ---------------------------------------------------------
APP_ROOT = Path(__file__).resolve().parents[2]
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
def now_iso() -> str: return datetime.utcnow().replace(microsecond=0).isoformat()
def ss_path(site: str, tag: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(SS_DIR / f"{ts}_{site}_{tag}.png")
def save_last_error(site: str, kind: str, message: str, screenshot: Optional[str] = None, extra: Optional[dict] = None):
    payload = {"timestamp": now_iso(), "site": site, "kind": kind, "message": message, "screenshot": screenshot}
    if extra: payload.update(extra)
    with open(ERR_JSON, "w", encoding="utf-8") as f: json.dump(payload, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------
# 通貨/価格ユーティリティ
# ---------------------------------------------------------
_CURRENCY_SIGNS = {"¥": "JPY", "￥": "JPY", "$": "USD", "€": "EUR", "£": "GBP", "₩": "KRW", "₫": "VND", "A$": "AUD", "C$": "CAD"}
_CURR_WORDS = {"JPY": ["JPY", "円"], "USD": ["USD"], "EUR": ["EUR"], "GBP": ["GBP"], "AUD": ["AUD"], "CAD": ["CAD"]}
def detect_currency(text: str, currency_hint: Optional[str] = None) -> str:
    if currency_hint: return currency_hint.upper()
    for sign, code in _CURRENCY_SIGNS.items():
        if sign in text: return code
    up = text.upper()
    for code, words in _CURR_WORDS.items():
        if any(w in up for w in words): return code
    return "UNKNOWN"
_PRICE_RE = re.compile(r"([0-9][0-9\.,\s]*)")
def to_number(price_text: str) -> Optional[int]:
    t = price_text.replace("\u00a0", " ")
    m = _PRICE_RE.search(t)
    if not m: return None
    digits = re.sub(r"[^\d]", "", m.group(1))
    return int(digits) if digits else None
def convert_price(price: int, currency: str, fx_to: Optional[str], fx_table: Dict[str, float]) -> Optional[float]:
    if not fx_to: return None
    fx_to, currency = fx_to.upper(), (currency or "").upper()
    if fx_to == "JPY":
        if currency == "JPY": return float(price)
        rate = fx_table.get(currency)
        return price * rate if rate else None
    return None

# ---------------------------------------------------------
# サイト設定モデル
# ---------------------------------------------------------
@dataclass
class SiteSelectors:
    search_open: List[str] = field(default_factory=list); search_input: List[str] = field(default_factory=list)
    search_submit: List[str] = field(default_factory=list); results_item: Optional[str] = None
    first_product_link: Optional[str] = None; pdp_title: List[str] = field(default_factory=list)
    pdp_price: List[str] = field(default_factory=list)
@dataclass
class SiteConfig:
    name: str; home_url: str
    domains: List[str] = field(default_factory=list)
    search_mode: str = "human"; search_template: Optional[str] = None
    wait_until: str = "domcontentloaded"; timeout_sec: int = 25
    currency_hint: Optional[str] = None; selectors: SiteSelectors = field(default_factory=SiteSelectors)
    force_ui_search: bool = False; notes: Optional[str] = None

# ---------------------------------------------------------
# デフォルトサイト（内蔵）
# ---------------------------------------------------------
def default_sites() -> List[SiteConfig]:
    return [
        SiteConfig(name="SSENSE", home_url="https://www.ssense.com/ja-jp", domains=["ssense.com"], search_template="https://www.ssense.com/ja-jp/search?q={q}", selectors=SiteSelectors(search_open=["a.mobile-header-search","i.fa-ssense-magnifier", "a[data-test='mobileNavigationSearchLink']", "button[aria-label='Open Search']"], search_input=["#search-form-input","input[data-testid='search-input']", "input[type='search']","input[name='q']"], search_submit=["#searchSubmitIcon","button[type='submit']"], results_item="a[href*='/product/']", first_product_link="a[href*='/product/']", pdp_title=["h1","h2#pdpProductNameText",".pdp-product-title__name"], pdp_price=["[data-test='pdpRegularPriceText']", ".product-price__sale", "span:has-text('¥')", "span:has-text('￥')"])),
        SiteConfig(name="BUYMA", home_url="https://www.buyma.com/", domains=["buyma.com"], search_template="https://www.buyma.com/r/-/search/?q={q}", selectors=SiteSelectors(search_input=["#search_txt","input.fab-search-txtarea","input#srchTxt"], search_submit=["form#search_form","button#srchBtn"], results_item="a[href*='/item/']", first_product_link="a[href*='/item/']", pdp_title=["h1[itemprop='name']","h1.product_title","h1"], pdp_price=["span.Price_Txt","#price",".product_price .Price_Txt","span[itemprop='price']"])),
        SiteConfig(name="FARFETCH", home_url="https://www.farfetch.com/", domains=["farfetch.com"], search_template="https://www.farfetch.com/shopping/men/items.aspx?q={q}", selectors=SiteSelectors(results_item="a[data-testid='productCard-link'], a[href*='/shopping/']", first_product_link="a[data-testid='productCard-link'], a[href*='/shopping/']", pdp_title=["h1[data-tstid='product-name']", "h1", "span[itemprop='name']"], pdp_price=["[data-tstid='priceInfo-original']", "[data-tstid='priceInfo-onsale']", "p[data-tstid='priceInfo']", "span[data-tstid='current-price']"])),
        SiteConfig(name="MATCHES", home_url="https://www.matchesfashion.com/", domains=["matchesfashion.com"], search_template="https://www.matchesfashion.com/intl/search?text={q}", selectors=SiteSelectors(results_item="a[href*='/products/']", first_product_link="a[href*='/products/']", pdp_title=["h1","h1[data-test='pdp-title']","div[data-test='pdp-title']"], pdp_price=["span[data-test='pdp-price']", "div.prices span.price", "span.now", "span.was"])),
        SiteConfig(name="GENERIC_PDP", home_url="", domains=[], selectors=SiteSelectors(pdp_title=["h1", "title"], pdp_price=["meta[itemprop='price']", "span[itemprop='price']", "span:has-text('¥')"]), notes="--pdp-url"),
    ]

# ---------------------------------------------------------
# 設定レイヤー読み込み
# ---------------------------------------------------------
def _update_dict_recursive(d, u):
    for k, v in u.items():
        if isinstance(v, dict): d[k] = _update_dict_recursive(d.get(k, {}), v)
        else: d[k] = v
    return d
def _load_json_if_exists(path: Path) -> dict:
    if not path.exists(): return {}
    try:
        with path.open("r", encoding="utf-8") as f: return json.load(f)
    except Exception as e:
        logging.warning(f"Could not load/parse JSON {path.name}: {e}"); return {}
def _dict_to_siteconfig(d: dict) -> SiteConfig:
    from dataclasses import fields as field_iter
    base_obj = SiteConfig(name="", home_url="")
    full_dict = asdict(base_obj)
    full_dict = _update_dict_recursive(full_dict, d)
    sel_dict = asdict(SiteSelectors())
    sel_dict = _update_dict_recursive(sel_dict, full_dict.get("selectors", {}))
    full_dict["selectors"] = SiteSelectors(**sel_dict)
    valid_keys = {f.name for f in field_iter(SiteConfig)}
    final_dict = {k: v for k, v in full_dict.items() if k in valid_keys}
    return SiteConfig(**final_dict)
def load_config_sites() -> List[SiteConfig]:
    log = logging.getLogger("ConfigLoader"); loaded_files = []
    sites_dict_map = {s.name.upper(): asdict(s) for s in default_sites()}
    base_data = _load_json_if_exists(CFG_BASE)
    if base_data:
        loaded_files.append(str(CFG_BASE.relative_to(APP_ROOT)))
        for site_conf in base_data.get("sites", []):
            name = (site_conf.get("name") or "").upper()
            if name in sites_dict_map:
                sites_dict_map[name] = _update_dict_recursive(sites_dict_map[name], site_conf)
    overrides_data = _load_json_if_exists(CFG_OVR)
    if overrides_data:
        loaded_files.append(str(CFG_OVR.relative_to(APP_ROOT)))
        for site_conf in overrides_data.get("sites", []):
            name = (site_conf.get("name") or "").upper()
            if name in sites_dict_map:
                sites_dict_map[name] = _update_dict_recursive(sites_dict_map[name], site_conf)
    legacy_data = _load_json_if_exists(CFG_LEGACY)
    if legacy_data:
        loaded_files.append(str(CFG_LEGACY.relative_to(APP_ROOT)))
        for site_conf in legacy_data.get("sites", []):
            name = (site_conf.get("name") or "").upper()
            if name not in sites_dict_map:
                 sites_dict_map[name] = site_conf
                 log.info(f"Loaded '{name}' from legacy config as fallback.")
    log.info(f"Config loaded from: {', '.join(loaded_files) if loaded_files else 'defaults only'}")
    return [_dict_to_siteconfig(d) for d in sites_dict_map.values()]

# ---------------------------------------------------------
# スカウト本体
# ---------------------------------------------------------
class SupplierScout:
    def __init__(self, headful=False, log_level="INFO", sites: Optional[List[str]] = None,
                 attempts=2, manual_verify=False, manual_wait_sec=240,
                 fx_to: Optional[str]=None, fx_rates: Optional[str]=None,
                 fx_auto: bool=True, fx_ttl_hours: int=12,
                 slow_mo: Optional[int]=None, speed_up: bool=False):
        logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s: %(name)s: %(message)s")
        self.log = logging.getLogger("SupplierScout")
        self.headful = headful; self.attempts = attempts; self.manual = manual_verify; self.manual_wait_sec = manual_wait_sec
        self.slow_mo = slow_mo; self.speed_up = speed_up
        self.fx_to = fx_to.upper() if fx_to else None
        manual_table = parse_fx_rates_str(fx_rates or "")
        auto_table, fx_meta = get_fx_table_jpy(auto=fx_auto, ttl_hours=fx_ttl_hours, manual_table=manual_table)
        self.fx_table = auto_table if auto_table else manual_table; self.fx_meta = fx_meta
        self.log.info(f"FX init: to={self.fx_to}, source={self.fx_meta.get('source')}, asof={self.fx_meta.get('asof')}")
        all_sites = load_config_sites()
        if sites:
            allow = set(s.upper() for s in sites); self.sites = [s for s in all_sites if s.name.upper() in allow]
        else: self.sites = all_sites
        self.log.info(f"SupplierScout ready: headful={self.headful}, slow_mo={self.slow_mo or 'dynamic'}, speed_up={self.speed_up}, sites={[s.name for s in self.sites]}")

    def _human_wait_random(self, min_ms=250, max_ms=800):
        if self.slow_mo: time.sleep(self.slow_mo / 1000)
        else: time.sleep(random.randint(min_ms, max_ms) / 1000)

    def _accept_popups(self, page: Page):
        selectors = ["button:has-text('Accept All')", "button:has-text('すべて承認')", "button:has-text('Agree')", "button:has-text('同意する')", "button:has-text('I agree')", "button:has-text('同意')", "#onetrust-accept-btn-handler", "[aria-label='Close']", "[aria-label='閉じる']"]
        for sel in selectors:
            try:
                page.locator(sel).first.click(timeout=1500); self.log.info(f"Clicked popup: {sel}"); self._human_wait_random(500, 1000); return
            except Exception: pass

    def _block_resources(self, route: Route):
        """画像、フォント、動画、広告などをブロックする"""
        req = route.request
        rtype = req.resource_type
        url = req.url
        block_list = ["googletag", "doubleclick", "analytics", "adservice", "googlead", "facebook", "fbcdn"]
        if rtype in {"image", "media", "font"} or any(s in url for s in block_list):
            route.abort()
        else:
            route.continue_()

    def _setup_page(self, page: Page):
        """ページ共通のセットアップ（リソースブロックなど）"""
        if self.speed_up:
            page.route("**/*", self._block_resources)

    def _run_pdp_direct_core(self, browser: Browser, site: SiteConfig, pdp_url: str) -> Optional[dict]:
        page = None
        try:
            page = browser.new_page()
            self._setup_page(page)
            page.set_default_timeout(site.timeout_sec * 1000)
            self.log.info(f"[{site.name}] Navigating directly to PDP: {pdp_url}")
            page.goto(pdp_url, wait_until=site.wait_until)
            self._accept_popups(page)
            title = (self._first_text(page, site.selectors.pdp_title) or "").strip()
            price_raw = self._first_text(page, site.selectors.pdp_price) or page.inner_text("body")
            currency = detect_currency(price_raw or "", site.currency_hint)
            price = to_number(price_raw or "")
            if not title or not price: raise RuntimeError("could not parse title or price from PDP body")
            price_conv = convert_price(price, currency, self.fx_to, self.fx_table)
            self.log.info(f"[{site.name}] PDP OK: {currency} {price} — {title[:60]}")
            return {"site": site.name, "url": page.url, "title": title, "price": price, "currency": currency, "price_converted": price_conv, "fx_to": self.fx_to}
        finally:
            if page: page.close()

    def _run_site_core(self, context: BrowserContext, site: SiteConfig, query: str) -> Optional[dict]:
        page = context.new_page()
        self._setup_page(page)
        page.set_default_timeout(site.timeout_sec * 1000)
        try:
            if site.force_ui_search or not site.search_template:
                self.log.info(f"[{site.name}] Using Human-First UI Interaction flow.")
                page.goto(site.home_url, wait_until=site.wait_until)
                self._accept_popups(page); self._human_wait_random(1000, 2500)
                page.mouse.move(random.randint(300, 600), random.randint(300, 600))
                self._search_ui(page, site, query)
            else:
                self.log.info(f"[{site.name}] Using direct search template.")
                url = site.search_template.format(q=quote_plus(query))
                page.goto(url, wait_until=site.wait_until)
                self._accept_popups(page)

            link_sel = site.selectors.first_product_link or site.selectors.results_item
            if not link_sel: raise RuntimeError("no result link selector")
            self._wait_visible_any(page, [link_sel])
            first = page.locator(link_sel).first
            self._human_wait_random()
            first.click()

            page.wait_for_url(r"**/product/**", timeout=site.timeout_sec * 1000)
            page.wait_for_load_state("networkidle")

            self._accept_popups(page); page.screenshot(path=ss_path(site.name, "pdp"))
            title = (self._first_text(page, site.selectors.pdp_title) or "").strip()
            price_raw = self._first_text(page, site.selectors.pdp_price) or page.inner_text("body")
            if not title or not price_raw: raise RuntimeError("could not parse title or price from page")
            currency = detect_currency(price_raw or "", site.currency_hint)
            price = to_number(price_raw or "")
            if not price: raise RuntimeError("could not convert price text to number")

            price_conv = convert_price(price, currency, self.fx_to, self.fx_table)
            self.log.info(f"[{site.name}] OK: {currency} {price} ({'~'+str(int(price_conv))+' '+self.fx_to if price_conv else 'no-fx'}) — {title[:60]}")
            return {"site": site.name, "url": page.url, "title": title, "price": price, "currency": currency, "price_converted": price_conv, "fx_to": self.fx_to}
        finally: page.close()

    def run_site_with_context(self, context: BrowserContext, site: SiteConfig, query: str) -> dict:
        self.log.info(f"--- Running site: {site.name} (force_ui_search: {site.force_ui_search}) ---")
        for attempt in range(1, self.attempts + 1):
            try:
                self.log.info(f"[{site.name}] attempt {attempt} start")
                result = self._run_site_core(context, site, query)
                if result: return {"ok": True, "data": result}
            except PWTimeout as e:
                p = ss_path(site.name, f"timeout_attempt_{attempt}")
                save_last_error(site.name, "TIMEOUT", f"Attempt {attempt}: {e}", p)
                self.log.warning(f"[{site.name}] TIMEOUT on attempt {attempt}: {e.__class__.__name__} (screenshot: {p})")
            except Exception as e:
                p = ss_path(site.name, f"error_attempt_{attempt}")
                save_last_error(site.name, "ERROR", f"Attempt {attempt}: {e}", p)
                self.log.warning(f"[{site.name}] ERROR on attempt {attempt}: {e.__class__.__name__} (screenshot: {p})", exc_info=False)

        self.log.error(f"[{site.name}] giving up after {self.attempts} attempts.")
        return {"ok": False, "kind": "GAVE_UP", "message": f"Failed after {self.attempts} attempts."}

    def _search_ui(self, page, site: SiteConfig, query: str):
        for sel in site.selectors.search_open or []:
            try: page.locator(sel).first.click(timeout=2000); self._human_wait_random(); break
            except Exception: continue
        self._human_wait_random(500, 1000)
        input_sel_found = False
        for sel in site.selectors.search_input or []:
            try:
                loc = page.locator(sel).first
                loc.click(); self._human_wait_random()
                loc.fill(query, timeout=5000); self._human_wait_random()
                input_sel_found = True; break
            except Exception: continue
        if not input_sel_found: raise RuntimeError(f"[{site.name}] no usable search input")
        for sel in site.selectors.search_submit or []:
            try: page.locator(sel).first.click(timeout=1000); self._human_wait_random(); break
            except Exception: continue
        else: page.keyboard.press("Enter"); self._human_wait_random()
        page.wait_for_load_state(site.wait_until)
    def _first_text(self, page, selectors: List[str]) -> Optional[str]:
        for sel in selectors:
            try:
                loc = page.locator(sel).first
                if loc.count() > 0:
                    t = loc.inner_text(timeout=5000).strip()
                    if t: return t
            except Exception: continue
        for sel in selectors:
            try:
                html = page.locator(sel).first.inner_html(timeout=1000)
                text_content = re.sub('<[^<]+?>', '', html).strip()
                if text_content: return text_content
            except Exception: continue
        return None
    def _wait_visible_any(self, page, selectors: List[str], timeout_ms: int = 25000):
        last_err = None
        for sel in selectors:
            try: page.locator(sel).first.wait_for(state="visible", timeout=timeout_ms); return
            except Exception as e: last_err = e; continue
        raise PWTimeout(f"wait any visible failed: {selectors} ({last_err})")
    def _human_wait(self, site: SiteConfig, page):
        self.log.warning(f"[HUMAN-CHECK] 検知: 手動解除を待機（~{self.manual_wait_sec}秒）")
        remain = self.manual_wait_sec
        while remain > 0:
            self.log.warning(f"[HUMAN-CHECK] 継続中… 残り ~{remain} 秒")
            page.wait_for_timeout(5000); remain -= 5
            html = page.content() or ""
            if all(t not in html for t in ["長押し", "Press and hold", "I'm not a robot"]): return
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
                    if r: results.append(r)
                finally: browser.close()
            else:
                self.log.info(f"Using persistent profile: {USER_DATA_DIR}")
                context = pw.chromium.launch_persistent_context(str(USER_DATA_DIR), headless=not self.headful,
                    slow_mo=self.slow_mo or 0,
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
                    viewport={'width': 1920, 'height': 1080}, screen={'width': 1920, 'height': 1080},
                    locale='ja-JP', timezone_id='Asia/Tokyo', args=['--start-maximized', '--disable-blink-features=AutomationControlled'])
                context.add_init_script("Object.defineProperty(navigator, 'webdriver', { get: () => undefined });")
                try:
                    for site in self.sites:
                        if site.name.upper() == "GENERIC_PDP": continue
                        res = self.run_site_with_context(context, site, product)
                        if res["ok"]: results.append(res["data"])
                finally: context.close()

        if not results: return {"ok": False, "message": "No valid supplier found."}
        best_key = lambda x: (x.get("price_converted") if self.fx_to and x.get("price_converted") is not None else float("inf"), x.get("price", float("inf")))
        best = sorted(results, key=best_key)[0]
        return {"ok": True, "fx": self.fx_meta, "best": {**best, "timestamp": now_iso()}, "results": results}

# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description=__doc__)
    p.add_argument("product", help="検索用の製品キーワード"); p.add_argument("--sites", nargs="+", help="対象サイト（例: FARFETCH SSENSE）")
    p.add_argument("--headful", action="store_true", help="ヘッドあり起動（動作確認用）"); p.add_argument("--log-level", default="INFO", help="DEBUG/INFO/WARN/ERROR")
    p.add_argument("--attempts", type=int, default=2, help="各サイトの試行回数"); p.add_argument("--manual-verify", action="store_true", help="人間検証が出たら手動待機")
    p.add_argument("--manual-wait-sec", type=int, default=240, help="手動待機の最大秒数"); p.add_argument("--pdp-url", help="PDP直指定URL（手動突破後のURL）")
    p.add_argument("--fx-to", help="換算先通貨（JPY）"); p.add_argument("--fx-rates", help="手動レート 'USD=155,EUR=170'")
    p.add_argument("--fx-auto", dest="fx_auto", action="store_true", default=True, help="為替レートを自動取得 ※既定: ON")
    p.add_argument("--no-fx-auto", dest="fx_auto", action="store_false", help="為替レートの自動取得を無効化")
    p.add_argument("--fx-ttl-hours", type=int, default=12, help="為替キャッシュのTTL(時間)")
    p.add_argument("--no-startup-fx-refresh", action="store_true", help="起動時の為替自動更新を無効化")
    p.add_argument("--slow-mo", type=int, help="操作間の固定遅延(ms)。デバッグ用。")
    p.add_argument("--speed-up", action="store_true", help="画像/広告等をブロックして高速化")
    args = p.parse_args()
    if not args.no_startup_fx_refresh and os.getenv("FX_AUTO_STARTUP", "1") == "1":
        try:
            _, meta = get_fx_table_jpy(auto=True, ttl_hours=24, manual_table={}); print(f"[FX] Auto-refresh: asof={meta.get('asof')}, source={meta.get('source')}")
        except Exception: print("[FX] Auto-refresh skipped due to error")
    scout = SupplierScout(headful=args.headful, log_level=args.log_level, sites=args.sites, attempts=args.attempts, manual_verify=args.manual_verify, manual_wait_sec=args.manual_wait_sec, fx_to=args.fx_to, fx_rates=args.fx_rates, fx_auto=args.fx_auto, fx_ttl_hours=args.fx_ttl_hours, slow_mo=args.slow_mo, speed_up=args.speed_up)
    result = scout.run_once(args.product, pdp_url=args.pdp_url)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1

if __name__ == "__main__":
    sys.exit(main())
