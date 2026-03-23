# -*- coding: utf-8 -*-
# ==============================================================================
# File Name : app/agents/browser_use_agent.py
# Version   : 85.3.3 (CompatFix)
# Date (JST): 2025年10月25日 23:13
# Usage     : drop-in replacement for previous BrowserUseAgent
# ------------------------------------------------------------------------------
# 変更要旨
#  - (v85.3.2まで): SPA, 高速化, 堅牢化, リスク調整 適用済
#  - ★ 修正 (v85.3.3): 実行時エラー修正
#    - (Hotfix 1) NameError 対策: PlaywrightError の import を多段フォールバック化。
#    - (Hotfix 2) TypeError 対策: 外部パッチ(moncler_plp_recovery)側の
#                 evaluate 6引数エラーについて、修正要点をコメントに明記。
# ==============================================================================
from __future__ import annotations
import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, urlunparse, urlencode, parse_qsl, quote_plus
from dataclasses import dataclass, field # ★ 統合: サイズ選択ポリシーのために追加

# ★ 修正 (v85.3.3): NameError 対策 (Hotfix 1)
# --- Playwright imports (robust) ---
from playwright.async_api import async_playwright, Page, Browser, BrowserContext, Route, Locator
# Error クラスはバージョンで場所が揺れるので多段フォールバック
try:
    from playwright.async_api import Error as PlaywrightError  # 新しめ
except Exception:
    try:
        from playwright.sync_api import Error as PlaywrightError  # 一部環境
    except Exception:
        try:
            from playwright._impl._errors import Error as PlaywrightError  # 内部実装
        except Exception:
            class PlaywrightError(Exception):  # 最終手段: 通常の Exception として扱う
                pass
# --- End of robust imports ---

# ★ クリーンアップ (v84.3): TargetClosedError, ElementHandle は不要

from app.core.run_context import RunContext
from app.utils.visual_regression import compare_and_maybe_update
from app.models.result_models import DiscoveryResult
from app.extractors.product_info_extractor import extract_title_price
from app.agents.failure_analysis_agent import FailureAnalysisAgent
from app.agents.selector_discovery_agent import SelectorDiscoveryAgent

from app.utils.observability import (
    save_dom, count_selectors, save_raw_hrefs, write_fail_snapshot
    # ★ クリーンアップ (v85.2): 未使用の save_json を削除 (提案4)
)

# --- Moncler専用パッチ（存在すれば利用、無ければスキップして既存挙動を維持） ---
try:
    try:
        from app.agents.browser_use_moncler_patch import moncler_plp_recovery
    except Exception:
        from browser_use_moncler_patch import moncler_plp_recovery  # プロジェクト直下想定
except Exception:
    moncler_plp_recovery = None  # import失敗時はNoneのまま。呼び出し側でガード。

logger = logging.getLogger(__name__)
_LOCALE_SEG_RE = re.compile(r"^[a-z]{2}-[a-z]{2}$", re.IGNORECASE)

# ★ 修正 (v85.3.2): バリアント親対策で 5 -> 8 に増強
MAX_PDP_LINKS_TO_FOLLOW = 8
DEFAULT_PDP_PARALLEL_LIMIT = 2
OVERALL_PLP_BUDGET_MS_DEFAULT = 120000  # 120s watchdog

# ==============================================================
# ★ 統合: PLP/PDP 堅牢化・サイズ選択のための定義
# (v85.2 最終微修正 反映)
# ==============================================================

@dataclass
class PDPSizeSelectPolicy:
# ... (v85.1 既存コード: 変更なし) ...
    """
    PDPで価格が表示されない場合に試みるサイズ選択の自動化ポリシー
    """
    mode: str = "off"              # "off" | "first_instock" | "by_label"
    prefer_labels: List[str] = field(default_factory=list)  # ["M","02","38"] 等。mode == "by_label" のとき優先探索

# ★ 修正 (v85.3.1): ロケール繰り返し許容 (パッチ①A)
PRODUCT_URL_ALLOW_PATTERNS = re.compile(
    # 説明:
    #  - 任意個のロケール片 (en-int / ja-jp …) を許容
    #  - /product(s)?|/p|/pp|/prod|/shop/.../(p|product) を拾う
    r"/(?:(?:[a-z]{2}-[a-z]{2})/)*"      # ← ★ 修正 (v85.3.1): ロケール繰り返しを許容
    r"(?:product(?:s)?|p|pp|prod|shop/[^/]+/(?:p|product))/"
    , re.IGNORECASE
)

# ★ 修正 (v85.2): 'wechat' を具体的なドメインに置き換え (提案2)
EXTERNAL_BLOCKLIST_HOSTS = (
    "line.me", "instagram.com", "facebook.com", "youtube.com", "x.com",
    "tiktok.com", "wa.me",
    # "wechat" は曖昧なので代表的な配信ドメインを明示
    "wechat.com", "weixin.qq.com", "wx.qq.com",
    # 広告・計測系 (ログノイズ低減)
    "doubleclick.net", "criteo.com", "googletagmanager.com", "googleadservices.com",
    "google-analytics.com", "googlesyndication.com"
)

# ★ 磨き込み: メタタグ・構造化データを優先する順序に変更
PRICE_SELECTORS = [
# ... (v85.1 既存コード: 変更なし) ...
    # 1. Meta (最優先)
    "meta[property='product:price:amount']",
    "meta[itemprop='price']",
    # 2. 構造化データ
    "[itemprop=price]",
    "span:has(meta[itemprop='price'])",
    # 3. 汎用セレクタ (フォールバック)
    "[data-testid*=price]",
    "[class*=price]",
]

# ★ クリーンアップ (v84.3): meta タグを除外した「可視」待機用セレクタ (提案B)
VISIBLE_PRICE_SELECTORS = [s for s in PRICE_SELECTORS if not s.startswith("meta[")]

# PDPでサイズボタン（在庫あり）を探すためのセレクタリスト
SIZE_BUTTON_SELECTORS = [
# ... (v85.1 既存コード: 変更なし) ...
    # よくあるサイズボタン（Moncler含む想定）
    "button[aria-disabled='false'][data-size]",
    "button[aria-disabled='false'][data-testid*='size']",
    "button:not([disabled])[data-size]",
    "li[data-size] button:not([disabled])",
    "button[class*='size']:not([disabled])",
    "[role='radiogroup'] [role='radio'][aria-checked='false']",
    # ★ 堅牢化 (v85.0): aria-pressed / aria-selected を追加 (提案4)
    "[aria-pressed='false'][class*='size']",
    "[aria-selected='false'][class*='size']",
]

def is_same_origin(url: str, base: str) -> bool:
# ... (v85.1 既存コード: 変更なし) ...
    """ 2つのURLが同一オリジン（スキーム+ホスト）か判定 """
    try:
        u = urlparse(url)
        b = urlparse(base)
        return (u.scheme, u.netloc) == (b.scheme, b.netloc)
    except Exception:
        return False

# ★ ブロッカー修正 (v85.1): ホスト名誤爆防止ヘルパー (提案2)
def _is_blocked_host(host: str) -> bool:
# ... (v85.1 既存コード: 変更なし) ...
    host = host.lower().strip(".")
    for bad in EXTERNAL_BLOCKLIST_HOSTS:
        bad = bad.lower()
        if host == bad or host.endswith("." + bad):
            return True
    return False

def looks_like_product_url(url: str) -> bool:
    """ URLが製品ページ(PDP)らしいか判定 """
    try:
        u = urlparse(url)

        # ★ 修正 (v85.2): 'blob:', 'data:' スキームを追加 (提案3)
        if u.scheme in {"mailto", "tel", "javascript", "blob", "data"}:
            return False

        if not u.path or u.path == "/":
# ... (v85.1 既存コード: 変更なし) ...
            return False

        # ★ ブロッカー修正 (v85.1): URL全体 ではなく _is_blocked_host (ホスト) をチェック (提案2)
        host = u.netloc.lower()
        if _is_blocked_host(host):
# ... (v85.1 既存コード: 変更なし) ...
            return False

        # 許可パターンにマッチするか
        return bool(PRODUCT_URL_ALLOW_PATTERNS.search(u.path)) # ★ 修正 (v85.3.1): 'shop/', 複数ロケール対応
    except Exception:
        return False
# ==============================================================
# (ここまで 統合定義)
# ==============================================================

def _unpack_vrt(res, *, threshold: Optional[float] = None):
# ... (v85.1 既存コード: 変更なし) ...
    if isinstance(res, dict):
        d = float(res.get("diff_percent", 0.0))
        ok = bool(res.get("is_ok", (threshold is None or d <= float(threshold or 0.0))))
        return {"diff_percent": d, "is_ok": ok}
    if isinstance(res, (tuple, list)):
        if not res:
            return None
        return _unpack_vrt(res[0], threshold=threshold)
    if isinstance(res, (int, float)):
        d = float(res)
        ok = (threshold is None) or (d <= float(threshold or 0.0))
        return {"diff_percent": d, "is_ok": ok}
    return None

def _is_candidate_pdp(url: str, allowed_domain: Optional[str]) -> bool:
# ... (v85.1 既存コード: 変更なし) ...
    # ★ 注意: この関数は _collect_pdp_links (v83) で使われるが、
    # 堅牢化ロジック (looks_like_product_url) で代替・強化される。
    try:
        u = urlparse(url)
        if allowed_domain and not u.netloc.endswith(allowed_domain):
            return False
        path = (u.path or "").lower()
        # ★ 修正 (v85.3.2): 保守性のため re.I を追加
        return bool(re.search(r"/product(s)?/|/p/|/pp/", path, flags=re.I))
    except Exception:
        return False

def _dedupe_keep_order(items: List[str]) -> List[str]:
# ... (v85.1 既存コード: 変更なし) ...
    return list(dict.fromkeys([i for i in (items or []) if i]))

class BrowserUseAgent:
    def __init__(self, runtime_kwargs: Optional[Dict[str, Any]] = None):
# ... (v85.1 既存コード: 変更なし) ...
        self.runtime_kwargs = runtime_kwargs or {}
        self.analysis_agent = FailureAnalysisAgent(runtime_kwargs=self.runtime_kwargs)
        self.discovery_agent = SelectorDiscoveryAgent(runtime_kwargs=self.runtime_kwargs)
        self.logger = logger

    # ---------- settings ----------
    def _resolve_run_settings(self, site_config: Dict[str, Any]) -> Dict[str, Any]:
# ... (v85.1 既存コード: 変更なし) ...
        ds = site_config.get("discovery_settings", {}) or {}
        vrt = ds.get("vrt", {}) or {}
        enable_har = self.runtime_kwargs.get("force_har", False) or self.runtime_kwargs.get("enable_har", ds.get("enable_har", True))
        enable_trace = self.runtime_kwargs.get("force_trace", False) or self.runtime_kwargs.get("enable_trace", ds.get("enable_trace", True))

        settings = { # ★ 統合: settings 変数を明示的に辞書として初期化
            "timeout_sec": self.runtime_kwargs.get("timeout_sec") or ds.get("timeout_sec", 60),
            "headless": self.runtime_kwargs.get("headless", True),
            "slow_mo": self.runtime_kwargs.get("slow_mo", 0),
            "viewport": ds.get("viewport"),
            "user_agent": ds.get("user_agent"),
            "extra_http_headers": ds.get("extra_http_headers"),
            "enable_har": enable_har,
            "enable_trace": enable_trace,
            "enable_video": self.runtime_kwargs.get("enable_video", ds.get("enable_video", False)),
            "enable_locale_escape": bool(ds.get("enable_locale_escape", True)),
            "overall_plp_budget_ms": int(ds.get("overall_plp_budget_ms", OVERALL_PLP_BUDGET_MS_DEFAULT)),
            "pdp_parallel_limit": int(ds.get("pdp_parallel_limit", DEFAULT_PDP_PARALLEL_LIMIT)),
            "pdp_retry_once": bool(ds.get("pdp_retry_once", True)), # ★ 堅牢化 (v84.3): この設定を活用
            # VRT
            "enable_visual_regression_check": bool(ds.get("enable_visual_regression_check", False)),
            "vrt_scope": (vrt.get("scope") or "none").lower(),
            "vrt_threshold": float(vrt.get("threshold", 0.02)),
            "vrt_hard_fail_threshold": float(vrt.get("hard_fail_threshold", 0.05)),
            "vrt_fail_on_hard_threshold": bool(vrt.get("fail_on_hard_threshold", True)),
            "vrt_baseline_dir": vrt.get("baseline_dir"),
            "vrt_plp_selector": vrt.get("plp_selector") or "full_page",
            "vrt_pdp_selector": vrt.get("pdp_selector") or "full_page",
            "vrt_auto_update_baseline": bool(vrt.get("auto_update_baseline", False)),
            "vrt_save_failed_diff_only": bool(vrt.get("save_failed_diff_only", True)),
            # selectors/wait
            "wait_for_selectors": _dedupe_keep_order(ds.get("wait_for_selectors") or []),
            "wait_until": ds.get("wait_until") or "domcontentloaded",
            "plp_scroll_rounds": int(ds.get("plp_scroll_rounds", 10)),
            "extra_block_routes": _dedupe_keep_order(ds.get("extra_block_routes") or []),
        }

        # ★ 磨き込み (v84.2): pdp_price_wait_ms を設定に追加
        settings["pdp_price_wait_ms"] = int(ds.get("pdp_price_wait_ms", 4000))

        # ★ 統合: site_config(Dict) から PDPSizeSelectPolicy オブジェクトを読み込み、settings(Dict) に格納
        try:
# ... (v85.1 既存コード: 変更なし) ...
            pdp_policy_cfg = ds.get("pdp_size_select_policy", {})
            if not isinstance(pdp_policy_cfg, dict):
                pdp_policy_cfg = {}
            settings["pdp_size_select_policy"] = PDPSizeSelectPolicy(
                mode=pdp_policy_cfg.get("mode", "off"),
                prefer_labels=pdp_policy_cfg.get("prefer_labels", [])
            )
        except Exception as e:
            logger.warning(f"Could not parse PDPSizeSelectPolicy from site_config: {e}. Defaulting to 'off'.")
            settings["pdp_size_select_policy"] = PDPSizeSelectPolicy()

        return settings

    # ---------- time budget helpers ----------
    @staticmethod
    def _start_watchdog(budget_ms: int) -> Tuple[float, int]:
# ... (v85.1 既存コード: 変更なし) ...
        return time.monotonic(), int(budget_ms)

    @staticmethod
    def _time_left_ms(start_t: float, budget_ms: int) -> int:
# ... (v85.1 既存コード: 変更なし) ...
        used = int((time.monotonic() - start_t) * 1000)
        remain = budget_ms - used
        return max(0, remain)

    @staticmethod
    def _slice_timeout_ms(left_ms: int, cap_ms: int) -> int:
# ... (v85.1 既存コード: 変更なし) ...
        return max(500, min(left_ms, cap_ms))

    # ---------- safe wait ----------
    async def safe_wait_selector(self, page: Page, selector: str, *, timeout_ms: int, state: str = "visible") -> bool:
# ... (v85.1 既存コード: 変更なし) ...
        if not page or (hasattr(page, "is_closed") and page.is_closed()):
            return False
        try:
            await page.wait_for_selector(selector, state=state, timeout=timeout_ms)
            return True
        # ★ 修正 (v85.3.3): 堅牢化された PlaywrightError を使用 (Hotfix 1)
        except (asyncio.CancelledError, PlaywrightError):
            return False
        except Exception:
            return False

    # ---------- UI helpers ----------
    async def _kill_overlays(self, page: Page) -> None:
# ... (v85.1 既存コード: 変更なし) ...
        try:
            await page.evaluate("""
              (() => {
                const sels = [
                  '.overlay','.backdrop','.modal-backdrop','#onetrust-banner-sdk','.cookie-banner',
                  '[aria-modal="true"]','.cmp-ui-overlay','.cmp-modal','.drawer--open'
                ];
                document.querySelectorAll(sels.join(',')).forEach(el => el.remove());
                const b = document.body; if (b) { b.classList.remove('modal-open','locked','no-scroll','overflow-hidden'); b.style.overflow=''; }
                const html=document.documentElement; if (html) { html.style.overflow=''; html.classList.remove('no-scroll','overflow-hidden'); }
              })();
            """)
        except Exception:
            pass

    # >>> PATCH: 強化版（リトライ＋検出拡張＋スクロール）
    async def _click_continue_shopping_if_present(self, page: Page, site_config: Dict[str, Any]) -> bool:
# ... (v85.1 既存コード: 変更なし) ...
        ui = (site_config.get("selectors") or {}).get("ui") or {}
        candidates = _dedupe_keep_order((ui.get("continue_shopping") or []) + [
            "a:has-text('CONTINUE SHOPPING')",
            "button:has-text('CONTINUE SHOPPING')",
            "[role='button']:has-text('CONTINUE SHOPPING')",
            "text=/\\bCONTINUE\\s+SHOPPING\\b/i"
        ])
        for round_ in range(3):
            try:
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
            except Exception:
                pass
            for sel in candidates:
                try:
                    el = page.locator(sel).first
                    if await el.count() > 0 and await el.is_visible():
                        await el.click(timeout=3000)
                        try:
                            await page.wait_for_load_state("domcontentloaded", timeout=3000)
                        except Exception:
                            pass
                        return True
                except Exception:
                    continue
            await page.wait_for_timeout(1200)
        return False

    async def _accept_cookies_if_present(self, page: Page, site_config: Dict[str, Any]) -> bool:
# ... (v85.1 既存コード: 変更なし) ...
        ui = (site_config.get("selectors") or {}).get("ui") or {}
        candidates = _dedupe_keep_order((ui.get("cookie_accept") or []) + [
            "#onetrust-accept-btn-handler",
            "button:has-text('ACCEPT ALL')",
            "button:has-text('CONTINUE WITHOUT ACCEPTING')",
            "button[aria-label*='Accept' i]",
        ])
        for sel in candidates:
            try:
                node = page.locator(sel).first
                if await node.count() > 0 and await node.is_visible():
                    await node.click(timeout=3000); await asyncio.sleep(0.2); return True
            except Exception:
                continue
        return False

    async def _footer_switch_to_english(self, page: Page, site_config: Dict[str, Any]) -> bool:
# ... (v85.1 既存コード: 変更なし) ...
        ui = (site_config.get("selectors") or {}).get("ui") or {}
        candidates = _dedupe_keep_order((ui.get("footer_language") or []) + [
            "footer a:has-text('ENGLISH')",
            "footer a[href*='en-int']",
            "footer a:has-text('EN')"
        ])
        try: await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
        except Exception: pass
        for sel in candidates:
            try:
                el = page.locator(sel).first
                if await el.count() > 0 and await el.is_visible():
                    await el.click(timeout=3000); await asyncio.sleep(0.2)
                    try: await page.wait_for_load_state("domcontentloaded")
                    except Exception: pass
                    return True
            except Exception:
                continue
        try: await page.evaluate("window.scrollTo(0, 0);")
        except Exception: pass
        return False

    async def _dismiss_geo_modal(self, page: Page) -> None:
# ... (v85.1 既存コード: 変更なし) ...
        for sel in [
            "text=STAY HERE","text=REMAIN HERE","text=REMAIN IN ENGLISH","text=CONTINUE SHOPPING",
            "text=ショッピングを続ける"
        ]:
            try:
                el = page.locator(sel).first
                if await el.count() > 0 and await el.is_visible():
                    await el.click(timeout=3000); break
            except Exception:
                continue

    # ---------- Moncler: locale helpers ----------
    def _normalize_to_en_int_url(self, url: str) -> str:
# ... (v85.1 既存コード: 変更なし) ...
        u = urlparse(url)
        path = (u.path or "/").replace("//","/")
        seg = [s for s in path.split("/") if s]

        # ★ 修正 (v85.3.2): 安全なロケール除去（リスク①対応）
        # 1. 先頭のロケール連続だけを除去
        i = 0
        while i < len(seg) and _LOCALE_SEG_RE.match(seg[i] or ""):
            i += 1
        seg = seg[i:]

        # 2. 直後に現れる重複 'en-int' だけ掃除
        seg = [s for s in seg if s.lower() != "en-int"]

        norm = "/en-int/" + "/".join(seg)
        if not norm.endswith("/"):
            norm += "/"
        q = dict(parse_qsl(u.query))
        q["forceLocale"] = "en-int"
        q.setdefault("shipToCountry","GB")
        return urlunparse((u.scheme, u.netloc, norm, u.params, urlencode(q), u.fragment))

    def _safe_en_int_url(self, home: str, path_or_full: str) -> str:
# ... (v85.1 既存コード: 変更なし) ...
        try:
            base = home or "https://www.moncler.com/en-int"
            raw = urljoin(base, path_or_full)
            return self._normalize_to_en_int_url(raw)
        except Exception:
            return self._normalize_to_en_int_url(path_or_full)

    async def _force_en_int(self, page: Page) -> None:
# ... (v85.1 既存コード: 変更なし) ...
        try:
            await page.context.add_cookies([
                {"name":"moncler-shipping-country","value":"GB","domain":".moncler.com","path":"/"},
                {"name":"moncler-shipping-language","value":"en","domain":".moncler.com","path":"/"},
            ])
        except Exception:
            pass
        try:
            fixed = self._normalize_to_en_int_url(page.url)
            if fixed != page.url:
                await page.goto(fixed, wait_until="domcontentloaded")
                try: await page.wait_for_load_state("networkidle", timeout=1500)
                except Exception: pass
        except Exception:
            pass

    async def _light_404_recover(self, page: Page, site_config: Dict[str, Any]) -> bool:
# ... (v85.1 既存コード: 変更なし) ...
        try:
            if await page.locator("text=It’s not here!, text=It's not here!").first.count() > 0:
                await self._click_continue_shopping_if_present(page, site_config)
            tpl = (site_config.get("discovery_settings", {}).get("url_templates") or {})
            recovery_path = tpl.get("official_catalog") or tpl.get("official_catalog_alt") or tpl.get("fallback_url")
            if recovery_path:
                url = self._safe_en_int_url(site_config.get("home_url",""), recovery_path)
                await page.goto(url, wait_until="domcontentloaded")
                await self._click_continue_shopping_if_present(page, site_config)  # >>> PATCH: 404直後も叩く
                try: await page.wait_for_load_state("networkidle", timeout=1500)
                except Exception: pass
                return True
        except Exception as e:
            logger.warning(f"[Recovery] 404 fallback error: {e}")
        return False

    # ---------- PLP materialize ----------
    async def _ensure_plp_materialized(self, page: Page, site_config: Dict[str, Any], settings: Dict[str, Any],
# ... (v85.1 既存コード: 変更なし) ...
                                       *, start_t: float, budget_ms: int) -> bool:
        pdp_cfg = (site_config.get("selectors") or {}).get("pdp", {}) or {}
        tile_selectors = _dedupe_keep_order(
            (pdp_cfg.get("pdp_link_selectors") or []) +
            (pdp_cfg.get("plp_container_selectors") or []) +
            ["a[data-product-url]", "[data-product-url]", "[data-qa='product-tile']",
             ".product-card", ".c-product-card", ".c-product-tile", "[data-testid*='product' i]"]
        )
        tile_selector_str = ", ".join(tile_selectors)
        target_min_tiles = 12
        max_scroll_attempts = int(max(settings.get("plp_scroll_rounds", 10), 10))

        for attempt in range(max_scroll_attempts):
# ... (v85.1 既存コード: 変更なし) ...
            left_ms = self._time_left_ms(start_t, budget_ms)
            if left_ms <= 0:
                self.logger.warning("[Materialize] Timed out.")
                return False

            # ★ 修正 (v85.2.5): 小刻みスクロール＋軽いネット待機（パッチ②）
            try:
                # 小刻みに数回スクロール → 画像/タイルのIO観測を起動させる
                for _ in range(6):
                    await page.evaluate("window.scrollBy(0, Math.floor(window.innerHeight*0.6))")
                    await page.wait_for_timeout(160)
                # 無限スクロール型に薄く効く idle 待機（短め）
                try:
                    await page.wait_for_load_state("networkidle", timeout=800)
                except Exception:
                    pass
            except Exception as e:
                self.logger.warning(f"[Materialize] Scroll failed on attempt {attempt + 1}: {e}")
                # (v85.2.5) スクロール失敗時はループを抜ける
                break

            try:
                count = await page.locator(tile_selector_str).count()
                self.logger.info(f"[Materialize] Attempt {attempt + 1}/{max_scroll_attempts}, found {count} tiles.")
                if count >= target_min_tiles:
# ... (v85.1 既存コード: 変更なし) ...
                    self.logger.info(f"[Materialize] Success: Found {count} tiles (>= {target_min_tiles}).")
                    return True
            except Exception as e:
                self.logger.warning(f"[Materialize] Could not count tiles on attempt {attempt + 1}: {e}")

        final_count = await page.locator(tile_selector_str).count()
# ... (v85.1 既存コード: 変更なし) ...
        if final_count > 0:
            self.logger.warning(f"[Materialize] Finished attempts, found {final_count} tiles (< {target_min_tiles}), but proceeding as non-empty.")
            return True

        self.logger.error("[Materialize] Failed: No product tiles found after all scroll attempts.")
# ... (v85.1 既存コード: 変更なし) ...
        return False

    # ==============================================================
    # ★ 統合: PDPサイズ自動選択ロジック (クラスメソッドとして追加)
# ... (v85.1 既存コード: 変更なし) ...
    # (v85.0 堅牢化 反映)
    # ==============================================================

    # ★ 堅牢化 (v85.0): Locator ベースに完全移行 (提案5)
    async def _read_price_or_none(self, page: Page) -> Optional[str]:
# ... (v85.1 既存コード: 変更なし) ...
        """
        現在のページから価格情報を読み取る。
        meta タグと可視テキストの両方を探索する。
        ★ v85.0: Locator ベース (nth) に移行。
        ★ v85.2.7: デタッチ耐性のため、要素ごとのtry/exceptを追加。（パッチ②）
        """
        try:
            for sel in PRICE_SELECTORS: # ★ 磨き込み: 優先順位付けされたセレクタを使用
                loc = page.locator(sel)
                count = await loc.count()
                if count == 0:
                    continue

                for i in range(count):
                    el = loc.nth(i)

                    # ★ 修正 (v85.2.7): 要素ごとの try/except でデタッチ耐性を強化
                    try:
                        # evaluate に e && を追加し、タグ取得の堅牢性を向上
                        tag_eval = await el.evaluate("e => e && e.tagName")
                        if not tag_eval:
                            continue # タグが取得できない要素はスキップ

                        tag = tag_eval.lower()
                        content = None

                        # meta タグの場合は content 属性、それ以外は inner_text
                        if tag == "meta":
                            content = await el.get_attribute("content")
                        else:
                            content = await el.inner_text()

                    except Exception:
                        # デタッチされた要素や、inner_textが取れない要素はスキップ
                        continue

                    # content が None でなく、中身があるか
                    if content:
                        content_stripped = content.strip()
                        # 価格らしき文字列（数字を含む）か簡易チェック
                        if content_stripped and re.search(r'\d', content_stripped):
                            # ★ 堅牢化 (v85.0): info -> debug (提案6)
                            self.logger.debug(f"Price found via selector '{sel}' (nth={i}): {content_stripped}")
                            return content_stripped

        except Exception as e:
            # ★ 堅牢化 (v85.0): info -> warning
            self.logger.warning(f"Error: Exception during _read_price_or_none (outer loop): {e}")

        # ★ 堅牢化 (v85.0): info -> debug (提案6)
        self.logger.debug("Price string not found (_read_price_or_none).")
        return None

    # ★ 堅牢化 (v84.3): .or_ 非対応フォールバック (提案E)
    async def _click_size_to_reveal_price(self, page: Page, policy: PDPSizeSelectPolicy, settings: Dict[str, Any]) -> bool:
# ... (v85.1 既存コード: 変更なし) ...
        """
        ポリシーに基づき、サイズボタンの自動クリックを試みる。
        ★ v84.3 修正: .or_ 非対応フォールバック追加。
        ★ v85.0 修正: ログレベル変更 (提案6)
        Returns: クリックが成功したか
        """
        if policy.mode == "off":
            return False

        # === ★ Locator ベースのロジック (v84.2) ===

        # 1. Locator として候補を収集
        locators: List[Locator] = []
        for sel in SIZE_BUTTON_SELECTORS: # ★ 堅牢化 (v85.0): 拡張されたセレクタリスト (提案4)
            loc = page.locator(sel)
            # count() が 0 より大きいかチェック
            try:
                if await loc.count() > 0:
                    locators.append(loc)
            except Exception:
                continue # タイムアウトや無効なセレクタをスキップ

        # 2. すべての候補を一つの Locator に統合
        if not locators:
            # ★ 堅牢化 (v85.0): info -> debug (提案6)
            self.logger.debug("No size buttons found to click (Locator).")
            return False

        buttons = locators[0]
        try:
            # ★ 堅牢化 (v84.3): .or_ がない古い環境のためのフォールバック (提案E)
            for loc in locators[1:]:
                # ★ 必須バグ修正 (v84.2): .union(loc) -> .or_(loc)
                buttons = buttons.or_(loc)
        except AttributeError:
            # .or_ が存在しない (AttributeError)
            self.logger.warning("Locator.or_() not found. Falling back to selector string union. (Old Playwright?)")
            try:
                selector_union = ", ".join(SIZE_BUTTON_SELECTORS)
                buttons = page.locator(selector_union)
            except Exception as e_fb:
                self.logger.error(f"Fallback locator strategy failed: {e_fb}")
                return False

        count = 0
        try:
            count = await buttons.count()
        except Exception as e:
            self.logger.warning(f"Error counting unified size buttons: {e}")

        if count == 0:
            # ★ 堅牢化 (v85.0): info -> debug (提案6)
            self.logger.debug("No size buttons found to click (count=0 after union/or_).")
            return False

        # 3. ラベル取得ヘルパ (Locator.nth(i) を対象)
        async def btn_label_at(i: int) -> str:
# ... (v85.1 既存コード: 変更なし) ...
            b = buttons.nth(i)
            try:
                txt  = (await b.inner_text() or "").strip()
                data = await b.get_attribute("data-size")
                aria = await b.get_attribute("aria-label")
                return (data or aria or txt or "").strip()
            except Exception:
                return ""

        clicked = False

        # 4. by_label 優先 (Locator.nth(i) に対して反復)
        if policy.mode == "by_label" and policy.prefer_labels:
# ... (v85.1 既存コード: 変更なし) ...
            prefer = [s.strip().upper() for s in policy.prefer_labels if s.strip()]
            for i in range(count):
                try:
                    b = buttons.nth(i)
                    if not await b.is_visible(): # OK: Locator method
                        continue
                    if await b.is_disabled(): # OK: Locator method
                        continue
                    lab = (await btn_label_at(i)).upper()
                    if lab and any(lab == L for L in prefer):
                        # ★ 堅牢化 (v85.0): info -> debug (提案6)
                        self.logger.debug(f"Clicking preferred size by label: {lab}")
                        await b.click()
                        clicked = True
                        break
                except Exception:
                    continue # 要素がデタッチされた場合など

        # 5. first_instock / by_labelの残り
        if not clicked and policy.mode in ("first_instock", "by_label"):
# ... (v85.1 既存コード: 変更なし) ...
            for i in range(count):
                try:
                    b = buttons.nth(i)
                    if not await b.is_visible(): # OK
                        continue
                    if await b.is_disabled(): # OK
                        continue

                    aria_dis = await b.get_attribute("aria-disabled")
                    if aria_dis == "true":
                        continue
                    txt = (await b.inner_text() or "").lower()
                    if "out of stock" in txt or "在庫なし" in txt:
                        continue

                    # ★ 堅牢化 (v85.0): info -> debug (提案6)
                    self.logger.debug(f"Clicking first available size (label: {await btn_label_at(i)})")
                    await b.click()
                    clicked = True
                    break
                except Exception:
                    continue # 要素がデタッチされた場合など

        # === ★ 修正ここまで ===

        if clicked:
# ... (v85.1 既存コード: 変更なし) ...
            try:
                # ★ 堅牢化 (v85.0): info -> debug (提案6)
                self.logger.debug("Waiting for price update after size click...")
                await page.wait_for_load_state("networkidle", timeout=2000)
            except Exception:
                # ★ 堅牢化 (v85.0): info -> debug (提案6)
                self.logger.debug("networkidle timeout, proceeding...")

            try:
                # ★ 磨き込み (v84.2): 待機時間を settings から取得
                wait_ms = int(settings.get("pdp_price_wait_ms", 4000))

                # ★ 修正 (v85.3.2): VISIBLE_PRICE_SELECTORS が空の場合のフォールバック（リスク③対応）
                sel = ", ".join(VISIBLE_PRICE_SELECTORS) or "[itemprop=price],[class*=price],[data-testid*=price]"

                await page.wait_for_selector(sel, state="visible", timeout=wait_ms)

                # ★ 堅牢化 (v85.0): info -> debug (提案6)
                self.logger.debug(f"Price selector appeared (visible) within {wait_ms}ms.")
            except Exception:
                # ★ 堅牢化 (v85.0): info -> debug (提案6)
                self.logger.debug("Price selector did not become visible after click (timeout).")
                pass

            await page.wait_for_timeout(500)

        return clicked

    async def _extract_price_with_size_option(self, page: Page, settings: Dict[str, Any]) -> Optional[str]:
# ... (v85.1 既存コード: 変更なし) ...
        """
        PDPから価格を抽出する。
        失敗した場合、ポリシーに基づきサイズ選択を試み、再抽出する。
        ★ v85.0 修正: ログレベル変更 (提案6)
        Returns: 価格テキスト or None
        """
        # まずは素直に読む
        price = await self._read_price_or_none(page)
        if price:
            return price

        policy = settings.get("pdp_size_select_policy", PDPSizeSelectPolicy())
        if policy.mode == "off":
            return None # サイズ選択がオフなら、ここで諦める

        # ★ 堅牢化 (v85.0): info -> debug (提案6)
        self.logger.debug("Price not found initially, attempting size selection...")
        # オプションに応じてサイズクリック → 再読込
        # ★ 磨き込み (v84.2): settings を _click_size_to_reveal_price に渡す
        if await self._click_size_to_reveal_price(page, policy, settings):
            price = await self._read_price_or_none(page)
            if price:
                # ★ 堅牢化 (v85.0): info -> debug (提案6)
                self.logger.debug("Price found after size selection.")
                return price
            else:
                # ★ 堅牢化 (v85.0): info -> debug (提案6)
                self.logger.debug("Price NOT found even after size selection.")

        # それでも無理なら None
        return None

    # ---------- PDP extraction ----------
    # ★ 堅牢化 (v85.0): タイムアウト引数を追加 (提案1)
    async def _extract_from_pdp(self, page: Page, url: str, *, settings: Dict[str, Any], close_after: bool = True, timeout_ms_override: Optional[int] = None) -> Optional[Dict[str, Any]]:
# ... (v85.1 既存コード: 変更なし) ...
        # ★ 統合: このメソッドの内部を、サイズ選択ロジックを呼び出すように変更
        try:
            self.logger.info(f"[PDP] Navigating to: {url}")

            # ★ 堅牢化 (v85.0): タイムアウトを引数で制御 (提案1)
            goto_timeout = timeout_ms_override or 30000

            if page.url != url:
                await page.goto(url, wait_until="domcontentloaded", timeout=goto_timeout)

            await self._kill_overlays(page)
            await self._click_continue_shopping_if_present(page, self.runtime_kwargs.get("site_config") or {})  # >>> PATCH

            if settings.get("enable_visual_regression_check") and "pdp" in (settings.get("vrt_scope") or ""):
                await self._perform_vrt(page, "pdp", settings)

            # --- ★ 統合: 新しいサイズ選択ロジック呼び出し (v84) ---
            # 1. まず価格を（サイズ選択ロジック経由で）試みる
            price_text = await self._extract_price_with_size_option(page, settings) # ★ 必須修正: この関数が Locator ベースで動作する

            # 2. 価格が見つかった場合のみ、残りの情報（タイトル等）を取得
            if price_text:
                # extract_title_price は価格以外の情報（タイトル等）も取得するため、
                # 価格が見つかった後に呼び出す（ただし、価格は上書きする）

                # ★ 修正 (v85.2.4): 抽出器がNoneを返した場合に備え、or {} でガード
                data = await extract_title_price(page) or {}

                data["price"] = price_text # サイズ選択で得た、より信頼できる価格で上書き
                data["url"] = page.url
                # extract_title_price が返す辞書の形式に合わせる (title, price, currency, url)
                # extract_title_price がラッパーであり、内部で extract_product_info を呼ぶ前提
                return data
            else:
                # サイズ選択を試みても価格が見つからなかった
                self.logger.warning(f"[PDP] Price not found (even after size attempts) at: {url}")
                return None
            # --- 統合ロジックここまで ---

        except Exception as e:
# ... (v85.1 既存コード: 変更なし) ...
            self.logger.error(f"[PDP] Extraction error at {url}: {e}", exc_info=True)
            return None
        finally:
            if close_after:
# ... (v85.1 既存コード: 変更なし) ...
                try:
                    if not page.is_closed():
                        await page.close()
                except Exception:
                    pass

    # ★ 堅牢化 (v84.3): pdp_retry_once ロジック (提案F)
    # ★ 堅牢化 (v85.0): タイムアウト予算管理 (提案1)
    async def _extract_from_pdp_list(self, page: Page, context: BrowserContext, site: str, query: str,
# ... (v85.1 既存コード: 変更なし) ...
                                     pdp_links: List[str], site_config: Dict, settings: Dict, run_context: RunContext,
                                     *, start_t: float, budget_ms: int) -> DiscoveryResult:
        original_plp_url = page.url
        # ★ 修正 (v85.3.2): MAX_PDP_LINKS_TO_FOLLOW (8) を使用
        limited_urls = pdp_links[:MAX_PDP_LINKS_TO_FOLLOW]

        left_ms = self._time_left_ms(start_t, budget_ms)
# ... (v85.1 既存コード: 変更なし) ...
        if left_ms <= 0:
            raise ValueError("Timed out before PDP extraction (watchdog).")

        sem = asyncio.Semaphore(int(settings.get("pdp_parallel_limit", 2)))

        # ★ 堅牢化 (v85.0): タイムアウト計算用のデフォルト (提案1)
        # 90秒か、残り予算の少ない方
        default_worker_cap_ms = min(90000, max(15000, left_ms))

        async def worker(u: str):
# ... (v85.1 既存コード: 変更なし) ...
            # ★ 堅牢化 (v84.3): リトライロジックのため、p は worker スコープで管理 (提案F)
            p: Optional[Page] = None
            try:
                # ★ 堅牢化 (v85.0): ワーカーごとに残り時間を確認 (提案1)
                left_ms_worker = self._time_left_ms(start_t, budget_ms)
                if left_ms_worker <= 2000: # 2秒以下なら実行不能
                    raise asyncio.TimeoutError("Not enough time left in budget for PDP worker.")

                # このワーカー用のタイムアウトをスライス
                worker_timeout = self._slice_timeout_ms(left_ms_worker, cap_ms=default_worker_cap_ms)

                async with sem:
                    p = await context.new_page()
                    # ★ 堅牢化 (v85.1): ワーカーにデフォルトタイムアウトを設定 (提案4)
                    p.set_default_timeout(worker_timeout)

                    # ★ 統合: _extract_from_pdp がサイズ選択ロジックを含む
                    # close_after=False にし、finally で閉じる
                    # ★ 堅牢化 (v85.0): スライスしたタイムアウトを渡す
                    item = await self._extract_from_pdp(p, u, settings=settings, close_after=False, timeout_ms_override=worker_timeout)

                    # ★ 堅牢化 (v84.3): リトライロジック (提案F)
                    if item is None and settings.get("pdp_retry_once", True):
# ... (v85.1 既存コード: 変更なし) ...

                        # ★ 堅牢化 (v85.0): リトライ前にも時間確認 (提案1)
                        left_ms_retry = self._time_left_ms(start_t, budget_ms)
                        if left_ms_retry <= 2000:
                            self.logger.warning(f"[PDP Worker] Retrying once for {u}... but no time left. Skipping.")
                            return None # タイムアウト

                        self.logger.info(f"[PDP Worker] Retrying once for {u}...")
# ... (v85.1 既存コード: 変更なし) ...
                        await asyncio.sleep(1.0) # 短い待機
                        try:
                            # リトライ用のタイムアウトをスライス
                            retry_timeout = self._slice_timeout_ms(left_ms_retry - 1000, cap_ms=default_worker_cap_ms)

                            # ページをリロード (goto)
                            await p.goto(u, wait_until="domcontentloaded", timeout=retry_timeout)
                            # 再度抽出を試みる
                            item = await self._extract_from_pdp(p, u, settings=settings, close_after=False, timeout_ms_override=retry_timeout)
                        except Exception as retry_e:
                            self.logger.warning(f"PDP worker retry failed for {u}: {retry_e}")

                    return item

            except Exception as e:
# ... (v85.1 既存コード: 変更なし) ...
                self.logger.warning(f"PDP worker failed for {u}: {e}")
                return e # 例外をそのまま返す
            finally:
                # ★ 堅牢化 (v84.3): p が存在し、閉じられていなければ必ず閉じる
# ... (v85.1 既存コード: 変更なし) ...
                if p and not p.is_closed():
                    try:
                        await p.close()
                    except Exception:
                        pass


        items = await asyncio.gather(*(worker(u) for u in limited_urls), return_exceptions=True)
# ... (v85.1 既存コード: 変更なし) ...
        valid_items = [it for it in items if isinstance(it, dict) and it]

        if not valid_items:
            # ★ 磨き込み: 例外メッセージを明確化 (v84.1)
            raise ValueError("Found PDP links but price extraction failed after size-selection and retry attempts.")

        # ★ ブロッカー修正 (v85.1): キーワード引数修正のため run_context.save_json を使用 (提案1)
        run_context.save_json("plp_extracted_items.json", {"extracted_data": valid_items})
        # ★ ブロッカー修正 (v85.1): キーワード引数を修正 (提案1)
        return DiscoveryResult(ok=True, site=site, query=query, message=f"PLP extracted {len(valid_items)} items",
                               evidence={"extracted_data": valid_items, "final_url": original_plp_url})

    # ---------- PDP links collection ----------
    # >>> PATCH: 深掘り版 (v83)
    async def _collect_pdp_links(self, page: Page, site_config: Dict, settings: Dict, run_context: RunContext) -> List[str]:
# ... (v85.1 既存コード: 変更なし) ...
        # ★ 統合: v83 の深掘りロジック + ユーザー提示の堅牢化フィルタリング
        target_url = page.url
        selectors_cfg = (site_config.get("selectors", {}) or {}).get("pdp", {}) or {}
        container_sels: List[str] = selectors_cfg.get("plp_container_selectors", []) or []
        link_sels: List[str] = selectors_cfg.get("pdp_link_selectors", []) or []

        # ★ v85.2.3 メモ: 以下の link_sels 構築は、サイト別チューニングや将来の拡張用。
        # 現在の主要なリンク抽出は、後続の scope.evaluate(...) (JavaScript) 内で実行される。
        for extra in [
# ... (v85.1 既存コード: 変更なし) ...
            "a[data-product-url]", "[data-product-url]", "[data-product-href]",
            "a[href^='/en-int/p/']", "a[href*='/en-int/products/']", "[role='link'][data-href]",
            "[data-qa='product-tile'] a[href]", "a[data-qa='product-card-link']",
            "a[data-testid*='product' i][href]", "a[href*='/products/']", "a[href*='/p/']",
            ".product-card a[href]", "[data-qa='product-tile'] [data-href]",
            "[data-testid*='product' i][data-href]", ".product-card [data-href]",
            "[data-qa='product-tile'] [onclick]", "[data-testid*='product' i][onclick]", ".product-card [onclick]",
        ]:
            if extra not in link_sels:
                link_sels.append(extra)

        for cont in (container_sels or []):
# ... (v85.1 既存コード: 変更なし) ...
            await self.safe_wait_selector(page, cont, timeout_ms=2000, state="visible")

        try:
# ... (v85.1 既存コード: 変更なし) ...
            for _ in range(3):
                await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                await page.wait_for_timeout(300)
        except Exception:
            pass

        # ★ 修正 (v85.2.5): スコープ拡張（パッチ①）
        # main領域に限定してノイズ低減 (v83 既存) -> 拡張
        scope = page.locator("main, [role='main'], #main, #app, body")

        try:
# ... (v85.1 既存コード: 変更なし) ...
            hrefs: List[str] = await scope.evaluate("""
              (node) => {
                const area = node || document;
                const out = [];
                const push = (u) => { if (u && typeof u === 'string') out.push(u); };

                // 1) 直接/親経由の a / data-href / data-product-url
                area.querySelectorAll("[data-product-url],[data-product-href],[data-href],a[href]").forEach(el=>{
                  const a = el.closest("a") || el;
                  const cand = a.getAttribute("href")
                    || a.getAttribute("data-href")
                    || a.getAttribute("data-product-url")
                    || a.getAttribute("data-product-href");
                  if (cand) push(cand);
                });

                // 2) onclick による遷移（location.href / assign / window.location / document.location / history.pushState）
                area.querySelectorAll("[onclick]").forEach(el=>{
                  const oc = el.getAttribute("onclick") || "";

                  // 従来
                  const m1 = oc.match(/(?:location\.(?:href|assign)|window\.location|document\.location)\s*=\s*['"]([^'"]+)['"]/i);
                  if (m1 && m1[1]) push(m1[1]);

                  // ★ 追加 (v85.3.0): pushState('...', '...', '/xxx') （パッチ 3）
                  const m2 = oc.match(/history\.pushState\s*\(\s*[^,]*,\s*[^,]*,\s*['"]([^'"]+)['"]\s*\)/i);
                  if (m2 && m2[1]) push(m2[1]);
                });

                // 3) JSON-LD から URL
                area.querySelectorAll("script[type='application/ld+json']").forEach(s=>{
                  try {
                    const data = JSON.parse(s.textContent || "null");
                    const arr = Array.isArray(data) ? data : [data];

                    // ★ 修正 (v85.2.6): offers 配列対応（パッチ②）
                    const pushAny = (v) => { if (v && typeof v === "string") push(v); };

                    arr.forEach(d=>{
                      if (!d || typeof d !== "object") return;

                      // メインの URL / @id
                      pushAny(d.url || d['@id']);

                      // offers (配列またはオブジェクト)
                      if (Array.isArray(d.offers)) {
                        d.offers.forEach(o => { pushAny(o && (o.url || o['@id'])); });
                      } else if (d.offers && typeof d.offers === "object") {
                        pushAny(d.offers.url || d.offers['@id']);
                      }

                      // itemListElement (既存ロジック)
                      if (d.itemListElement && Array.isArray(d.itemListElement)){
                        d.itemListElement.forEach(it=>{
                          if (it && it.item && (it.item.url || it.item['@id'])) {
                            pushAny(it.item.url || it.item['@id']);
                          }
                        });
                      }
                    });
                  } catch(e){}
                });

                return out.filter(Boolean);
              }
            """)
        except Exception:
            hrefs = []

        allowed_domain = site_config.get("allowed_domain")
        abs_urls = [urljoin(target_url, h) for h in hrefs if h]

        await save_raw_hrefs(run_context, abs_urls, name="raw_hrefs_raw_abs")

        abs_urls = [u for u in abs_urls if not u.endswith("#") and not u.lower().startswith("javascript:")]

        # ★ 修正 (v85.3.1): リンク収集(B): "素の a[href]" スイープを常時実行 (パッチ①B)
        for q in ["a[href*='/en-int/p/']", "a[href*='/p/']"]:
            try:
                loc = page.locator(q)
                c = await loc.count()
                for i in range(min(c, 200)):
                    href = await loc.nth(i).get_attribute("href")
                    if href:
                        abs_urls.append(urljoin(target_url, href))
            except Exception:
                pass

        # ★ 修正 (v85.3.1): リンク収集(B): "data-*" 属性からの収集 (パッチ①B)
        for sel in ["[data-pdp-url]", "[data-url]", "[data-product-url]", "[data-href]"]:
            try:
                loc = page.locator(sel)
                c = await loc.count()
                for i in range(min(c, 120)):
                    v = await loc.nth(i).get_attribute("data-pdp-url") \
                        or await loc.nth(i).get_attribute("data-url") \
                        or await loc.nth(i).get_attribute("data-product-url") \
                        or await loc.nth(i).get_attribute("data-href")
                    if v:
                        abs_urls.append(urljoin(target_url, v))
            except Exception:
                pass

        # v83 既存のフィルタリング
        cand_v83 = [u for u in abs_urls if _is_candidate_pdp(u, allowed_domain)]

        # ★ 統合: ユーザー提示の堅牢化ロジック (is_same_origin, looks_like_product_url) を
        # v83の深掘り抽出結果(abs_urls) に対して適用する
        cand_robust = []
        base = page.url
        for u in abs_urls:
            if not is_same_origin(u, base):
                continue
            if not looks_like_product_url(u): # ★ 堅牢化 (v85.3.1): フィルタ強化
                continue
            cand_robust.append(u)

        # v83のロジック(cand_v83) と 堅牢化ロジック(cand_robust) の両方（和集合）を採用し、重複排除する
        final_candidates = _dedupe_keep_order(cand_robust + cand_v83)

        # ★ 修正 (v85.2.5): 緊急スイープを v85.3.1 で常時実行に格上げしたため、ここのブロックは削除

        # ★ 修正 (v85.2.2): IndentationError を修正
        if not final_candidates and cand_v83:
# ... (v85.1 既存コード: 変更なし) ...
            self.logger.warning("[_collect_pdp_links] Robust filters (is_same_origin/looks_like) excluded all links. Check EXTERNAL_BLOCKLIST_HOSTS or PRODUCT_URL_ALLOW_PATTERNS.")
        elif not final_candidates:
            self.logger.warning("[_collect_pdp_links] No PDP links found after deep extraction and filtering.")

        return final_candidates


    # ---------- VRT ----------
    async def _perform_vrt(self, page: Page, scope: str, settings: Dict[str, Any]):
# ... (v85.1 既存コード: 変更なし) ...
        try:
            from pathlib import Path as _P
            site_name = self.runtime_kwargs.get("site") or "GENERIC"
            baseline_dir = _P(settings.get("vrt_baseline_dir") or f"app/visual_baselines/{site_name}")
            baseline_dir.mkdir(parents=True, exist_ok=True)
            sel = settings.get("vrt_plp_selector") if scope=="plp" else settings.get("vrt_pdp_selector")
            res = await compare_and_maybe_update(
                page=page, baseline_path=baseline_dir / f"{scope}.png", selector=sel,
                threshold=settings.get("vrt_threshold", 0.02), hard_fail_threshold=settings.get("vrt_hard_fail_threshold", 0.05),
                auto_update_baseline=settings.get("vrt_auto_update_baseline", False),
                save_failed_diff_only=settings.get("vrt_save_failed_diff_only", True))
            vrt_result = _unpack_vrt(res, threshold=settings.get("vrt_threshold", 0.02))
            if vrt_result and not vrt_result.get("is_ok", True):
                diff = float(vrt_result.get("diff_percent", 0.0))
                hard = diff > float(settings.get("vrt_hard_fail_threshold", 0.05))
                msg = f"[VRT][{scope.upper()}] diff={diff:.4f} (thr={settings.get('vrt_threshold', 0.02)})"
                if hard and settings.get("vrt_fail_on_hard_threshold", True) and scope == "pdp":
                    # ★ 修正 (v85.3.3): 堅牢化された PlaywrightError を使用 (Hotfix 1)
                    raise PlaywrightError(msg + f" HARD>{settings.get('vrt_hard_fail_threshold', 0.05)}")
                logger.warning(msg + (" HARD" if hard else ""))
        except Exception as vrt_e:
            logger.warning(f"[VRT][{scope.upper()}] skipped due to error: {vrt_e}")

    # ---------- context ----------
    def _build_context_options(self, settings: Dict[str, Any], run_context: RunContext) -> Dict[str, Any]:
# ... (v85.1 既存コード: 変更なし) ...
        ctx_opts: Dict[str, Any] = {}
        if settings.get("viewport"): ctx_opts["viewport"] = settings["viewport"]
        ctx_opts["locale"] = "en-GB"; ctx_opts["timezone_id"] = "UTC"
        headers = (settings.get("extra_http_headers") or {}).copy()
        headers.setdefault("Accept-Language","en-GB,en;q=0.8")
        ctx_opts["extra_http_headers"] = headers
        if settings.get("user_agent"): ctx_opts["user_agent"] = settings["user_agent"]
        if settings.get("enable_har"): ctx_opts["record_har_path"] = str(run_context.get_path("network.har"))
        if settings.get("enable_video"):
            videos_dir = run_context.get_path("videos"); Path(videos_dir).mkdir(parents=True, exist_ok=True)
            ctx_opts["record_video_dir"] = str(videos_dir)
        return ctx_opts

    # ★ 修正 (v85.2.7): extra_block_routes を「ホスト」と「グロブ」に分離（パッチ①）
    async def _setup_routes(self, context: BrowserContext, settings: Dict[str, Any]):

        base_routes = ["**/*onetrust.com/**","**/*cookielaw.org/**","**/*cookiepro.com/**"]
        extra = settings.get("extra_block_routes") or []

        # 1) グロブとホストを仕分け
        extra_hosts, extra_globs = [], []
        for x in extra:
            x = (x or "").strip()
            if not x:
                continue
            # 簡易判定: グロブ記号 or スラッシュを含むなら「パターン」
            if "*" in x or "/" in x:
                extra_globs.append(x)
            else:
                # ドメイン/サブドメインらしければホスト扱い
                extra_hosts.append(x.lstrip("."))

        # 2) ホストブロックリストを作成
        block_hosts = tuple(h.lower().lstrip(".") for h in EXTERNAL_BLOCKLIST_HOSTS) + tuple(h.lower() for h in extra_hosts)

        async def _abort(route: Route):
            """ (v85.2.3) ブロック対象ルートを安全に abort するハンドラ """
            try:
                await route.abort()
            except Exception:
                # 既に abort された場合や、コンテキストが閉じた場合のエラーを無視
                pass

        async def _locale_rewrite(route: Route):
            """
            自己完結型ハンドラ：
            1. ロケール書き換え
            2.「ホスト名」ブロック判定 (v85.2.7で extra_hosts を含む)
            3. fallback() / continue_()
            """
            try:
                req = route.request; url = req.url

                # 1) ドキュメントだけロケール書き換え
                if req.resource_type in ("document",) and "moncler.com" in url:

                    # ★ 修正 (v85.3.1): 検索/インデックス系パスは書換スキップ (パッチ②A)
                    pu = urlparse(url)
                    path_lower = (pu.path or "/").lower()
                    skip_paths = ("/search", "/account", "/customer", "/help", "/privacy", "/legal")

                    is_skippable = False
                    if path_lower == "/":
                        is_skippable = True
                    else:
                        for s in skip_paths:
                            # /search や /search/ で始まるパスを検知
                            if path_lower.startswith(s) or path_lower.startswith(f"{s}/"):
                                is_skippable = True
                                break

                    if not is_skippable:
                        fixed = self._normalize_to_en_int_url(url)
                        if fixed != url:
                            self.logger.info(f"[Route] URL normalized: {url} -> {fixed}")
                            await route.continue_(url=fixed); return
                    else:
                        self.logger.debug(f"[Route] Skipping locale normalization for index/search path: {path_lower}")

                # 2)「ホスト名」ブロック判定
                host = urlparse(url).netloc.lower().strip(".")
                if any(host == b or host.endswith("." + b) for b in block_hosts):
                    await route.abort(); return

                # 3) 可能なら次のハンドラへ
                try:
                    await route.fallback(); return
                except Exception:
                    # 旧環境: fallback() がない場合、続行
                    await route.continue_()

            except Exception as e:
                self.logger.warning(f"[Route] handler error: {e}")
                try:
                    # 予期せぬエラー時もリクエストは続行させる
                    await route.continue_()
                except Exception:
                    pass # コンテキスト閉じなどの二重エラーは無視

        # ★ 修正 (v85.3.2): 順序が意図的であることを明記（リスク④対応）
        # この順序は意図的:
        # 1. catch-all (ロケール書き換え / ホスト名ブロック / fallback) を先に登録
        await context.route("**/*", _locale_rewrite)

        # 2. 個別のグロブパターン (abort) を後から登録
        #    _locale_rewrite が fallback() を呼ぶと、これらの個別ルートが評価される
        for pat in base_routes + extra_globs:
            try:
                await context.route(pat, _abort)
            except Exception as e:
                logger.warning(f"Route block (glob) failed for {pat}: {e}")


    async def _setup_init_scripts(self, context: BrowserContext):
# ... (v85.1 既存コード: 変更なし) ...
        await context.add_init_script("""
          try { localStorage.setItem('a11y-contrast','off'); localStorage.setItem('high-contrast','off'); } catch(e){}
          Object.defineProperty(navigator, 'language', {get: () => 'en-GB'});
          Object.defineProperty(navigator, 'languages', {get: () => ['en-GB','en']});
          (function(){ const _rz=Intl.DateTimeFormat.prototype.resolvedOptions;
            Intl.DateTimeFormat.prototype.resolvedOptions=function(){const o=_rz.call(this); o.timeZone='UTC'; return o;}; })();
        """)

    # ---------- main ----------
    async def run(self, *, site: str, query: str, site_config: Dict[str, Any],
# ... (v85.1 既存コード: 変更なし) ...
                  run_context: RunContext, target_url: str, likely_plp: bool) -> DiscoveryResult:

        self.runtime_kwargs["site_config"] = site_config
        self.runtime_kwargs["site"] = site
        mode = (self.runtime_kwargs or {}).get("mode", "run").lower()

        settings = self._resolve_run_settings(site_config) # ★ 統合: この呼び出しでサイズ選択ポリシー/待機時間も読み込まれる
        timeout_ms = int(settings.get("timeout_sec", 60)) * 1000

        try:
# ... (v85.1 既存コード: 変更なし) ...
            instance_dir = Path(run_context.run_path).parent.parent
            learned_path = instance_dir / "sites" / site.upper() / "learned_selectors.json"
            if learned_path.exists():
                learned = json.loads(learned_path.read_text(encoding="utf-8"))
                sc_sel = site_config.setdefault("selectors", {}).setdefault("pdp", {})
                for k in ("price_selectors","title_selectors","pdp_link_selectors"):
                    v = learned.get(k)
                    if v:
                        sc_sel[k] = list(dict.fromkeys(list(v) + list(sc_sel.get(k, []))))
                logger.info(f"[LEARN] loaded and merged selectors: {learned_path}")
        except Exception as e:
            logger.warning(f"[LEARN] merge skipped: {e}")

        async with async_playwright() as p:
# ... (v85.1 既存コード: 変更なし) ...
            browser: Optional[Browser] = None
            context: Optional[BrowserContext] = None
            page: Optional[Page] = None
            try:
                browser = await p.chromium.launch(headless=settings.get("headless", True), slow_mo=settings.get("slow_mo", 0))
                context = await browser.new_context(**self._build_context_options(settings, run_context))
# ... (v85.1 既存コード: 変更なし) ...
                await self._setup_routes(context, settings)
                await self._setup_init_scripts(context)

                if settings.get("enable_trace"):
# ... (v85.1 既存コード: 変更なし) ...
                    await context.tracing.start(screenshots=True, snapshots=True, sources=True)

                if site.upper() == "MONCLER_OFFICIAL":
# ... (v85.1 既存コード: 変更なし) ...
                    await context.add_cookies([
                        {"name":"moncler-shipping-country","value":"GB","domain":".moncler.com","path":"/"},
                        {"name":"moncler-shipping-language","value":"en","domain":".moncler.com","path":"/"},
                    ])
                    logger.info("[Moncler] Pre-injected locale/country cookies.")

                page = await context.new_page()
                page.set_default_timeout(timeout_ms)

                await page.goto(target_url, wait_until="domcontentloaded")
                await self._accept_cookies_if_present(page, site_config)
# ... (v85.1 既存コード: 変更なし) ...
                await self._dismiss_geo_modal(page)
                await self._kill_overlays(page)
                await self._click_continue_shopping_if_present(page, site_config)   # >>> PATCH: 常時チェック
                try: await page.wait_for_load_state("domcontentloaded", timeout=800)
# ... (v85.1 既存コード: 変更なし) ...
                except Exception: pass
                await page.wait_for_timeout(120)

                if site.upper() == "MONCLER_OFFICIAL":
# ... (v85.1 既存コード: 変更なし) ...
                    try:
                        gate_links_count = await page.evaluate("() => document.querySelectorAll(\"a[href*='/en-']\").length")
                        if gate_links_count and gate_links_count >= 10:
                            logger.warning(f"[Moncler] Locale gate detected ({gate_links_count} links). Forcing navigation to PLP.")
                            fixed_url = "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB"
                            await page.goto(fixed_url, wait_until="domcontentloaded")
                            await self._click_continue_shopping_if_present(page, site_config)  # >>> PATCH
                            try: await page.wait_for_load_state("networkidle", timeout=2000)
                            except Exception: pass
                            await self._accept_cookies_if_present(page, site_config)
                    except Exception as gate_e:
                        logger.warning(f"[Moncler] Gate detection logic failed: {gate_e}")

                if await self._light_404_recover(page, site_config):
# ... (v85.1 既存コード: 変更なし) ...
                    await run_context.take_screenshot(page, "11_after_404_recovery")

                if settings.get("enable_locale_escape") and site.upper() == "MONCLER_OFFICIAL":
# ... (v85.1 既存コード: 変更なし) ...
                    await self._force_en_int(page)
                    await self._click_continue_shopping_if_present(page, site_config)  # >>> PATCH
                    await run_context.take_screenshot(page, "12_after_locale_escape")

                # --- VRTの直後、PLPフロー突入前に一度だけ Moncler 専用パッチを呼ぶ ---
                await run_context.take_screenshot(page, "20_pre_vrt_and_extraction")
# ... (v85.1 既存コード: 変更なし) ...
                if settings.get("enable_visual_regression_check") and "plp" in (settings.get("vrt_scope") or ""):
                    await self._perform_vrt(page, "plp", settings)

                if site.upper() == "MONCLER_OFFICIAL" and moncler_plp_recovery is not None:
# ... (v85.1 既存コード: 変更なし) ...
                    try:
                        # ★【重要】修正 (v85.3.3): 外部パッチ 'moncler_plp_recovery' 内部の
                        # Page.evaluate() 呼び出しが6引数になっており、TypeError の原因です。
                        # ユーザー指示の通り、evaluate の第2引数に全引数を {dict} で渡すよう、
                        # *外部ファイル* (browser_use_moncler_patch.py) を修正してください。
                        await moncler_plp_recovery(page, site_config, query)
                    except Exception as _e:
                        logger.warning(f"[MonclerPatch] skipped: {_e}")

                start_t, budget_ms = self._start_watchdog(settings.get("overall_plp_budget_ms", OVERALL_PLP_BUDGET_MS_DEFAULT))

                if mode == "learn":
# ... (v85.1 既存コード: 変更なし) ...
                    return await self._run_learning_flow(page, context, site, site_config, settings, run_context, start_t=start_t, budget_ms=budget_ms)

                if likely_plp:
# ... (v85.1 既存コード: 変更なし) ...
                    return await self._run_plp_flow(page, context, site, query, site_config, settings, run_context, start_t=start_t, budget_ms=budget_ms)
                else:
                    return await self._run_pdp_flow(page, site, query, settings, run_context)

            except Exception as e:
# ... (v85.1 既存コード: 変更なし) ...
                return await self._handle_run_failure(e, site, query, site_config, run_context, page)
            finally:
                try:
# ... (v85.1 既存コード: 変更なし) ...
                    if context and settings.get("enable_trace"):
                        await context.tracing.stop(path=run_context.get_path("trace.zip"))
                except Exception: pass
                if browser: await browser.close()

    async def _run_plp_flow(self, page: Page, context: BrowserContext, site: str, query: str,
# ... (v85.1 既存コード: 変更なし) ...
                            site_config: Dict, settings: Dict, run_context: RunContext,
                            *, start_t: float, budget_ms: int) -> DiscoveryResult:

        await self._ensure_plp_materialized(page, site_config, settings, start_t=start_t, budget_ms=budget_ms)
        try:
# ... (v85.1 既存コード: 変更なし) ...
            await save_dom(run_context, page, "plp_dom_initial_materialized")
            pdp_cfg_a = (site_config.get("selectors") or {}).get("pdp", {}) or {}
            await count_selectors(
                run_context, page,
                (pdp_cfg_a.get("pdp_link_selectors") or []) + (pdp_cfg_a.get("plp_container_selectors") or []),
                name="selector_counts_plp_initial"
            )
        except Exception as e:
            logger.warning(f"[Hook A1] Failed: {e}")

        pdp_links = await self._collect_pdp_links(page, site_config, settings, run_context) # ★ 統合: 堅牢化されたフィルタリングがここで実行される (v85.3.1)

        if not pdp_links:
# ... (v85.1 既存コード: 変更なし) ...
            try:
                # ★ 堅牢化 (v85.0): info -> debug (提案6)
                self.logger.debug("[Fallback] Trying header search UI...")
                did_search = await self._plp_header_search_fallback(page, query, site_config, settings, run_context, context, start_t=start_t, budget_ms=budget_ms)
                if did_search:
# ... (v85.1 既存コード: 変更なし) ...
                    await self._click_continue_shopping_if_present(page, site_config)  # >>> PATCH

                    # ★ 修正 (v85.2.5): 軽量マテリアライズ条件（アンカーが痩せてる時だけ）（パッチ③）
                    try:
                        anchors = await page.locator("a[href*='/en-int/p/'], a[href*='/p/']").count()
                    except Exception:
                        anchors = 0
                    if anchors < 6:
                        self.logger.debug(f"[Fallback] light materialize after search (anchors={anchors} < 6)")
                        await self._ensure_plp_materialized(page, site_config, settings, start_t=start_t, budget_ms=budget_ms)

                    try:
# ... (v85.1 既存コード: 変更なし) ...
                        await save_dom(run_context, page, "plp_dom_search_fallback")
                        pdp_cfg_a2 = (site_config.get("selectors") or {}).get("pdp", {}) or {}
                        await count_selectors(
                            run_context, page,
                            (pdp_cfg_a2.get("pdp_link_selectors") or []) + (pdp_cfg_a2.get("plp_container_selectors") or []),
                            name="selector_counts_after_search_fallback"
                        )
                    except Exception as e:
                        logger.warning(f"[Hook A3] Failed: {e}")

                    pdp_links = await self._collect_pdp_links(page, site_config, settings, run_context) # ★ 統合: 堅牢化 (v85.3.1)

                    if not pdp_links:
# ... (v85.1 既存コード: 変更なし) ...
                        self.logger.warning("[Fallback:header-search] PLP loaded, but no hrefs found. Attempting to click the first product card.")
                        new_page = await self._click_first_card_or_link(page, site_config, settings, context)
                        if new_page:
                            return await self._run_pdp_flow(new_page or page, site, query, settings, run_context)

                        # ★ 追加 (v85.3.0): new_page が無くても、PDP指標が見えていれば同タブで抽出に行く (パッチ 2)
                        try:
                            await page.wait_for_selector(", ".join(VISIBLE_PRICE_SELECTORS), state="visible", timeout=2000)
                            self.logger.warning("[Fallback:click-card] No navigation event, but PDP indicators appeared; trying same-tab PDP extraction.")
                            return await self._run_pdp_flow(page, site, query, settings, run_context)
                        except Exception:
                            pass

            except Exception as _e:
                self.logger.warning(f"[Fallback:header-search] failed: {_e}", exc_info=True)

        if pdp_links:
# ... (v85.1 既存コード: 変更なし) ...
            # ★ 統合: この呼び出し先がサイズ選択ロジック + リトライ (v84.3) + タイムアウトスライス (v85.0) を含む
            return await self._extract_from_pdp_list(page, context, site, query, pdp_links, site_config, settings, run_context, start_t=start_t, budget_ms=budget_ms)

        raise ValueError("All PDP attempts failed to extract price, even after all recovery attempts.")

    # >>> PATCH: 検索UI同期強化
    async def _plp_header_search_fallback(self, page, query: str, site_config, settings, run_context, context: BrowserContext, *, start_t: float, budget_ms: int) -> bool:
# ... (v85.1 既存コード: 変更なし) ...
        ui = (site_config.get("selectors") or {}).get("ui") or {}
        sel_open = _dedupe_keep_order(ui.get("search_open", []) + ["button[aria-label='Search']", "[aria-label*='Search' i]"])
        sel_input = _dedupe_keep_order(ui.get("search_input", []) + [
            "form[role='search'] input", "input[type='search']", "input[name='q']",
            "[data-testid*='search' i] input", "[role='search'] input", "dialog input[type='search']"
        ])
        sel_submit = _dedupe_keep_order(ui.get("search_submit", []) + ["form[role='search'] button[type='submit']"])

        try:
# ... (v85.1 既存コード: 変更なし) ...
            opened = False
            for s in sel_open:
                if self._time_left_ms(start_t, budget_ms) <= 0: break
                el = page.locator(s).first
                if await el.count() > 0:
                    await el.click(timeout=3000); opened = True; await asyncio.sleep(0.2)
                    await self.safe_wait_selector(page, "[role='search'], [data-overlay], dialog[open]", timeout_ms=5000, state="visible")
                    # ★ 堅牢化 (v85.0): info -> debug (提案6)
                    self.logger.debug(f"[Fallback:header-search] opened with '{s}'")
                    break
            if not opened:
# ... (v85.1 既存コード: 変更なし) ...
                await page.keyboard.press("/")
                await self.safe_wait_selector(page, "input[type='search']", timeout_ms=5000, state="visible")

            found_input = False
# ... (v85.1 既存コード: 変更なし) ...
            for s in sel_input:
                if self._time_left_ms(start_t, budget_ms) <= 0: break
                el = page.locator(s).first
                if await el.count() > 0 and await el.is_visible():
                    await el.fill(query, timeout=8000); found_input = True
                    # ★ 堅牢化 (v85.0): info -> debug (提案6)
                    self.logger.debug(f"[Fallback:header-search] filled '{query}' into '{s}'"); break
            if not found_input: raise ValueError("Input field not found")

            submitted = False
# ... (v85.1 既存コード: 変更なし) ...
            for s in sel_submit:
                if self._time_left_ms(start_t, budget_ms) <= 0: break
                el = page.locator(s).first
                if await el.count() > 0 and await el.is_enabled():
                    await el.click(timeout=5000); submitted = True
                    # ★ 堅牢化 (v85.0): info -> debug (提案6)
                    self.logger.debug(f"[Fallback:header-search] submitted with '{s}'"); break
            if not submitted:
                # ★ 堅牢化 (v85.0): info -> debug (提案6)
                await page.keyboard.press("Enter"); self.logger.debug("[Fallback] submitted with Enter key.")

            left_ms = self._time_left_ms(start_t, budget_ms)
# ... (v85.1 既存コード: 変更なし) ...
            if left_ms > 1000:
                await page.wait_for_load_state("domcontentloaded", timeout=min(left_ms, 15000))
                # ★ 仕上げ (v85.2.2): 最小限の可視要素待機を追加し、空PLPの確率を低減
                try:
                    await page.wait_for_selector("main, #main, [role='main']", state="visible", timeout=800)
                except Exception:
                    self.logger.debug("[Fallback] Optional main element wait timed out (800ms).")

            return True
        except Exception:
            self.logger.warning("[Fallback] UI search failed. Trying direct search URL as final fallback.")
# ... (v85.1 既存コード: 変更なし) ...
            try:
                search_url = f"https://www.moncler.com/en-int/search?q={quote_plus(query)}&forceLocale=en-int"
                await page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
                await self._click_continue_shopping_if_present(page, site_config)  # >>> PATCH

                # ★ 修正 (v85.2.1): 直行検索フォールバックでのマテリアライズを削除（高速化）
                # (v85.2.5) パッチ③は呼び出し元(_run_plp_flow)で実施するため、ここは削除のまま

                # ★ 仕上げ (v85.2.2): 最小限の可視要素待機を追加
                try:
                    await page.wait_for_selector("main, #main, [role='main']", state="visible", timeout=800)
                except Exception:
                    self.logger.debug("[Fallback] Optional main element wait (URL fallback) timed out (800ms).")

                return True
            except Exception as final_e:
                self.logger.error(f"[Fallback] Direct search URL navigation failed: {final_e}")
                return False

    # ★ 堅牢化 (v84.3): 競合タスクの例外吸収フォールバック (提案A)
    # ★ 堅牢化 (v85.0): about:blank 遅延ロード対策 (提案B) + ログレベル (提案6)
    # ★ 堅牢化 (v85.1): 同一タブ遷移時の wait_for_url (提案3)
    # ★ 修正 (v85.3.0): SPA ナビゲーション検知（パッチ 1）
    async def _click_and_capture_navigation(self, click_coro, page: Page, context: BrowserContext, *,
                                            url_regex: Optional[re.Pattern] = re.compile(r"/product[s]?/|/p/|/pp/", re.I),
                                            wait_state: str = "domcontentloaded", timeout_ms: int = 15000) -> Optional[Page]:

        # 既存のタスク
        popup_task = asyncio.create_task(context.wait_for_event("page", timeout=timeout_ms))
        same_tab_nav_task = asyncio.create_task(page.wait_for_event("framenavigated", timeout=timeout_ms))

        # ★ 追加 (v85.3.0): SPA でも勝てる第三・第四のタスク
        # 1) URL 変化（正規表現マッチ）待ち
        spa_url_task = None
        if url_regex:
            try:
                # 直前のURLを避けるため、小さく遅延を入れてから待機開始
                await asyncio.sleep(0)  # イベントループ1tick
                spa_url_task = asyncio.create_task(page.wait_for_url(url_regex, timeout=timeout_ms))
            except Exception:
                spa_url_task = None

        # 2) PDP指標（価格セレクタ or meta[itemprop=price]）可視化待ち
        # ★ 修正 (v85.3.2): VISIBLE_PRICE_SELECTORS が空の場合のフォールバック（リスク③対応）
        sel_spa = ", ".join(VISIBLE_PRICE_SELECTORS) or "[itemprop=price],[class*=price],[data-testid*=price]"
        spa_price_task = asyncio.create_task(
            page.wait_for_selector(sel_spa, state="visible", timeout=timeout_ms)
        )

        try:
            await click_coro()
        except Exception:
            for t in (popup_task, same_tab_nav_task, spa_url_task, spa_price_task):
                if t and not t.done(): t.cancel()
            return None

        # ★ 追加 (v85.3.0): 4レースの FIRST_COMPLETED
        tasks = {popup_task, same_tab_nav_task, spa_price_task}
        if spa_url_task: tasks.add(spa_url_task)

        try:
            done, pending = await asyncio.wait(tasks, timeout=timeout_ms/1000, return_when=asyncio.FIRST_COMPLETED)
            for t in pending: t.cancel()
            if not done:
                return None

            winner = next(iter(done))

            # 勝者が「新規ページ」ならそのまま
            if winner is popup_task:
                new_page = winner.result()
                self.logger.debug(f"[_click_and_capture] Winner: new page (popup)")
            else:
                # framenavigated / url-match / price-visible のどれでも「同タブでPDP化」と見なす
                new_page = page
                if winner is same_tab_nav_task:
                    self.logger.debug(f"[_click_and_capture] Winner: framenavigated (same page)")
                elif winner is spa_url_task:
                    self.logger.debug(f"[_click_and_capture] Winner: SPA URL match (same page)")
                elif winner is spa_price_task:
                    self.logger.debug(f"[_click_and_capture] Winner: SPA Price selector visible (same page)")

            # about:blank ガード等は既存どおり
            try:
                if new_page.url == "about:blank":
                    self.logger.debug("[_click_and_capture] New page is about:blank, waiting for domcontentloaded...")
                    await new_page.wait_for_load_state('domcontentloaded', timeout=1500)
            except Exception as e_blank:
                self.logger.debug(f"[_click_and_capture] Wait for about:blank load failed: {e_blank}")

            # 同タブ時も軽く待つ (v85.0)
            try:
                await new_page.wait_for_load_state(wait_state, timeout=max(500, timeout_ms // 10))
            except Exception:
                pass

            # URL 正規表現がある場合、改めて短く待つ（SPAでも大抵ここで確定）
            if url_regex:
                try:
                    await new_page.wait_for_url(url_regex, timeout=max(1000, timeout_ms // 4))
                    # ★ 修正 (v85.3.1): コメント追記 (パッチ④)
                    # URLだけ確定して可視セレクタが来ない場合でも、
                    # この時点で new_page が返るため、PDP抽出に進むことができる (SPA/遅延対応)
                except Exception as e_url_final:
                    self.logger.debug(f"[_click_and_capture] Final wait_for_url (non-fatal) failed: {e_url_final}")
                    pass

            return new_page

        except Exception as e_wait:
            self.logger.warning(f"[_click_and_capture] Navigation race failed: {e_wait}")
            return None # どのタスクも完了しなかったか、勝者の .result() で例外

        finally:
            for t in (popup_task, same_tab_nav_task, spa_url_task, spa_price_task):
                if t and not t.done(): t.cancel()


    async def _click_first_card_or_link(self, page: Page, site_config: Dict, settings: Dict, context: BrowserContext) -> Optional[Page]:
# ... (v85.1 既存コード: 変更なし) ...
        pdp = (site_config.get("selectors", {}).get("pdp") or {})
        link_sel = pdp.get("pdp_link_selectors", [])
        plp_boxes = pdp.get("plp_container_selectors", ["main", "section[role='main']", "#main", "[id*='product' i]", "[class*='product' i]"])
        block_ng = set(pdp.get("blocklist_href_substrings", ["/cart", "/wishlist", "javascript:void"]))
        url_pat = re.compile(r"/product[s]?/|/p/|/pp/", re.I)
        if link_sel:
            for s in link_sel:
                try:
                    # ★ 堅牢性/互換性 (v84.2): .all() -> .count() / .nth()
                    loc = page.locator(s)
                    count = await loc.count()
                    for i in range(count):
                        el = loc.nth(i)
                        # (既存ロジック再開)
                        href = (await el.get_attribute("href")) or (await el.get_attribute("data-href")) or ""
                        if href and not any(bad in href for bad in block_ng):
                            await el.scroll_into_view_if_needed()
                            # ★ 修正 (v85.3.0): SPA対応のナビゲーション検知を呼ぶ
                            newp = await self._click_and_capture_navigation(lambda: el.click(timeout=5000), page, context, url_regex=url_pat)
                            if newp: return newp
                except Exception: continue
        tile_selectors = [
# ... (v85.1 既存コード: 変更なし) ...
            "[data-qa='product-tile']", ".c-product-tile", ".product-card",
            "[data-testid*='product-card']", "article[data-product-id]"
        ]
        for box in plp_boxes:
            for tile_sel in tile_selectors:
                try:
                    # ★ 修正 (v85.2.2): 互換性の低いセレクタ (>> css=) を修正
                    card = page.locator(f"{box} {tile_sel}").first
                    if await card.count() == 0: continue
                    await card.scroll_into_view_if_needed()
                    for action in (lambda: card.click(timeout=5000),):
                        # ★ 修正 (v85.3.0): SPA対応のナビゲーション検知を呼ぶ
                        newp = await self._click_and_capture_navigation(action, page, context, url_regex=url_pat)
                        if newp: return newp
                except Exception: continue
        self.logger.warning("[Fallback:click-card] Could not find any clickable link or card.")
        return None

    async def _run_pdp_flow(self, page: Page, site: str, query: str, settings: Dict, run_context: RunContext) -> DiscoveryResult:
# ... (v85.1 既存コード: 変更なし) ...
        logger.info("[Mode] PDP (detail)")
        item = await self._extract_from_pdp(page, page.url, settings=settings, close_after=False) # ★ 統合: この呼び出し先がサイズ選択ロジックを含む
        try:
# ... (v85.1 既存コード: 変更なし) ...
            await save_dom(run_context, page, "pdp_dom")
        except Exception as e:
            logger.warning(f"[Hook C1] Failed: {e}")
        if not item:
            raise ValueError("Price not found on PDP.") # ★ 堅牢化 (v84.3): リトライは _run_plp_flow 側で処理される
        run_context.save_json("pdp_extracted_data.json", {"extracted_data": item})
        # ★ ブロッカー修正 (v85.1): キーワード引数を修正 (提案1)
        return DiscoveryResult(ok=True, site=site, query=query, message="PDP extracted",
                               evidence={"extracted_data": item, "final_url": item.get("url")})

    async def _handle_run_failure(self, e: Exception, site: str, query: str, site_config: Dict,
# ... (v85.1 既存コード: 変更なし) ...
                                  run_context: RunContext, page: Optional[Page]) -> DiscoveryResult:
        logger.error(f"Browser task failed (RunID: {run_context.run_id}): {e}", exc_info=True)
        final_url_on_fail = None
        try:
            if page and not page.is_closed(): final_url_on_fail = page.url
        except Exception: pass
        analysis = self.analysis_agent.analyze(error=e, site=site, site_config=site_config,
                                              html_content="", run_context=run_context)
        try:
            await write_fail_snapshot(run_context, page, final_url_on_fail, e, site_config)
        except Exception as hook_e:
            logger.warning(f"[Hook C2] Failed: {hook_e}")
        # ★ ブロッカー修正 (v85.1): キーワード引数を修正 (提案1)
        return DiscoveryResult(ok=False, site=site, query=query, message=str(e),
                               ai_analysis=analysis, evidence={"final_url": final_url_on_fail})

    # ---------- learning flow ----------
    async def _run_learning_flow(self, page: Page, context: BrowserContext, site: str, site_config: Dict,
# ... (v85.1 既存コード: 変更なし) ...
                                 settings: Dict, run_context: RunContext, *, start_t: float, budget_ms: int) -> DiscoveryResult:
        self.logger.info(f"[LEARN] Starting learning flow for site: {site}")
        await self._ensure_plp_materialized(page, site_config, settings, start_t=start_t, budget_ms=budget_ms)
        await save_dom(run_context, page, "learn_plp_dom_for_discovery")
        try:
# ... (v85.1 既存コード: 変更なし) ...
            discovered_selectors = await self.discovery_agent.discover(page=page, context=context, run_context=run_context)
            if not discovered_selectors:
                raise ValueError("SelectorDiscoveryAgent returned no selectors.")
            self.logger.info(f"[LEARN] Discovered selectors: {json.dumps(discovered_selectors, indent=2)}")
        except Exception as e:
            self.logger.error(f"[LEARN] Selector discovery failed: {e}", exc_info=True)
            # ★ ブロッカー修正 (v85.1): キーワード引数を修正 (提案1)
            return DiscoveryResult(ok=False, site=site, query="(learning)", message=f"Selector discovery failed: {e}")
        try:
            await self._save_learned_selectors(site, discovered_selectors, run_context)
        except Exception as e:
            self.logger.error(f"[LEARN] Failed to save learned selectors: {e}", exc_info=True)
            # ★ ブロッカー修正 (v85.1): キーワード引数を修正 (提案1)
            return DiscoveryResult(ok=False, site=site, query="(learning)", message=f"Failed to save learned selectors: {e}")
        # ★ ブロッカー修正 (v85.1): キーワード引数を修正 (提案1)
        return DiscoveryResult(ok=True, site=site, query="(learning)",
                               message="Successfully learned and saved new selectors.",
                               evidence={"learned_selectors": discovered_selectors})

    async def _save_learned_selectors(self, site: str, new_selectors: Dict[str, List[str]], run_context: RunContext) -> None:
# ... (v85.1 既存コード: 変更なし) ...
        try:
            instance_dir = Path(run_context.run_path).parent.parent
            site_dir = instance_dir / "sites" / site.upper()
            site_dir.mkdir(parents=True, exist_ok=True)
            learned_path = site_dir / "learned_selectors.json"
        except Exception:
            learned_path = Path(f"instance/sites/{site.upper()}/learned_selectors.json")
            learned_path.parent.mkdir(parents=True, exist_ok=True)

        if learned_path.exists():
# ... (v85.1 既存コード: 変更なし) ...
            try:
                existing_selectors = json.loads(learned_path.read_text(encoding="utf-8"))
                if not isinstance(existing_selectors, dict): existing_selectors = {}
            except (json.JSONDecodeError, TypeError):
                existing_selectors = {}
        else:
            existing_selectors = {}

        merged_selectors = {}
# ... (v85.1 既存コード: 変更なし) ...
        all_keys = set(existing_selectors.keys()) | set(new_selectors.keys())
        for key in all_keys:
            new_list = new_selectors.get(key, [])
            existing_list = existing_selectors.get(key, [])
            merged_list = _dedupe_keep_order(new_list + existing_list)
            if merged_list:
                merged_selectors[key] = merged_list

        learned_path.write_text(json.dumps(merged_selectors, indent=2, ensure_ascii=False), encoding="utf-8")
# ... (v85.1 既存コード: 変更なし) ...
        self.logger.info(f"[LEARN] Saved merged selectors to: {learned_path}")

