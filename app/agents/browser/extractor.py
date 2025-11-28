# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from playwright.async_api import BrowserContext, Page

from app.core.run_context import RunContext
from app.models.result_models import DiscoveryResult
from app.extractors.product_info_extractor import extract_title_price
from app.extractors.moncler_extractor import MonclerPDPExtractor
from app.agents.browser.session_manager import EXTERNAL_BLOCKLIST_HOSTS
from app.utils.observability import save_dom

logger = logging.getLogger(__name__)

PRODUCT_URL_ALLOW_PATTERNS = re.compile(
    r"/(?:(?:[a-z]{2}-[a-z]{2})/)*(?:product(?:s)?|p|pp|prod|shop/[^/]+/(?:p|product))/",
    re.IGNORECASE,
)

PRICE_SELECTORS = [
    "meta[property='product:price:amount']",
    "meta[itemprop='price']",
    "[itemprop=price]",
    "span:has(meta[itemprop='price'])",
    "[data-testid*=price]",
    "[class*=price]",
]
VISIBLE_PRICE_SELECTORS = [s for s in PRICE_SELECTORS if not s.startswith("meta[")]

SIZE_BUTTON_SELECTORS = [
    "button[aria-disabled='false'][data-size]",
    "button[aria-disabled='false'][data-testid*='size']",
    "button:not([disabled])[data-size]",
    "li[data-size] button:not([disabled])",
    "button[class*='size']:not([disabled])",
    "[role='radiogroup'] [role='radio'][aria-checked='false']",
    "[aria-pressed='false'][class*='size']",
    "[aria-selected='false'][class*='size']",
]

MAX_PDP_LINKS_TO_FOLLOW = 8
DEFAULT_PDP_PARALLEL_LIMIT = 2
OVERALL_PLP_BUDGET_MS_DEFAULT = 120000

PreparePageCallable = Optional[Callable[[Page], Awaitable[None]]]


@dataclass
class PDPSizeSelectPolicy:
    mode: str = "off"
    prefer_labels: List[str] = field(default_factory=list)


def _is_blocked_host(host: str) -> bool:
    host = host.lower().strip(".")
    for bad in EXTERNAL_BLOCKLIST_HOSTS:
        bad_norm = bad.lower()
        if host == bad_norm or host.endswith("." + bad_norm):
            return True
    return False


def looks_like_product_url(url: str) -> bool:
    from urllib.parse import urlparse

    try:
        u = urlparse(url)
        if u.scheme in {"mailto", "tel", "javascript", "blob", "data"}:
            return False
        if not u.path or u.path == "/":
            return False
        host = u.netloc.lower()
        if _is_blocked_host(host):
            return False
        return bool(PRODUCT_URL_ALLOW_PATTERNS.search(u.path))
    except Exception:
        return False


class BrowserExtractionService:
    def __init__(self, logger: logging.Logger, runtime_kwargs: Optional[Dict[str, Any]] = None) -> None:
        self.logger = logger
        self.runtime_kwargs = runtime_kwargs or {}
        self.moncler_extractor = MonclerPDPExtractor(logger)

    async def extract_single_pdp(
        self,
        *,
        page: Page,
        context: Optional[BrowserContext],
        site: str,
        query: str,
        settings: Dict[str, Any],
        run_context: RunContext,
        site_config: Dict[str, Any],
        target_url: str,
        prepare_page: PreparePageCallable = None,
    ) -> DiscoveryResult:
        item = await self._extract_from_pdp(
            page=page,
            url=target_url or page.url,
            context=context,
            site=site,
            settings=settings,
            site_config=site_config,
            prepare_page=prepare_page,
        )
        if not item:
            raise ValueError("Price not found on PDP.")

        try:
            await save_dom(run_context, page, "pdp_dom")
        except Exception as e:
            self.logger.warning(f"[Extractor] Failed to save PDP DOM: {e}")

        run_context.save_json("pdp_extracted_data.json", {"extracted_data": item})
        return DiscoveryResult(
            ok=True,
            site=site,
            query=query,
            message="PDP extracted",
            evidence={"extracted_data": item, "final_url": item.get("url")},
        )

    async def extract_from_pdp_list(
        self,
        *,
        page: Page,
        context: BrowserContext,
        site: str,
        query: str,
        pdp_links: List[str],
        site_config: Dict[str, Any],
        settings: Dict[str, Any],
        run_context: RunContext,
        start_t: float,
        budget_ms: int,
        prepare_page: PreparePageCallable = None,
    ) -> DiscoveryResult:
        original_plp_url = page.url
        limited_urls = pdp_links[:MAX_PDP_LINKS_TO_FOLLOW]
        left_ms = self._time_left_ms(start_t, budget_ms)
        if left_ms <= 0:
            raise ValueError("Timed out before PDP extraction (watchdog).")

        sem = asyncio.Semaphore(int(settings.get("pdp_parallel_limit", DEFAULT_PDP_PARALLEL_LIMIT)))
        default_worker_cap_ms = min(90000, max(15000, left_ms))

        async def worker(u: str):
            worker_page: Optional[Page] = None
            try:
                left_ms_worker = self._time_left_ms(start_t, budget_ms)
                if left_ms_worker <= 2000:
                    raise asyncio.TimeoutError("Not enough time left for PDP worker.")
                worker_timeout = self._slice_timeout_ms(left_ms_worker, cap_ms=default_worker_cap_ms)
                async with sem:
                    worker_page = await context.new_page()
                    worker_page.set_default_timeout(worker_timeout)
                    return await self._extract_from_pdp(
                        page=worker_page,
                        url=u,
                        context=context,
                        site=site,
                        settings=settings,
                        site_config=site_config,
                        timeout_override=worker_timeout,
                        prepare_page=prepare_page,
                    )
            except Exception as e:
                self.logger.warning(f"[PDP Worker] Failed for {u}: {e}")
                return None
            finally:
                if worker_page and not worker_page.is_closed():
                    try:
                        await worker_page.close()
                    except Exception:
                        pass

        items = await asyncio.gather(*(worker(u) for u in limited_urls), return_exceptions=False)
        valid_items = [it for it in items if isinstance(it, dict) and it]
        if not valid_items:
            raise ValueError("Found PDP links but price extraction failed after size-selection and retry.")

        run_context.save_json("plp_extracted_items.json", {"extracted_data": valid_items})
        return DiscoveryResult(
            ok=True,
            site=site,
            query=query,
            message=f"PLP extracted {len(valid_items)} items",
            evidence={"extracted_data": valid_items, "final_url": original_plp_url},
        )

    async def _extract_from_pdp(
        self,
        *,
        page: Page,
        url: str,
        context: Optional[BrowserContext],
        site: str,
        settings: Dict[str, Any],
        site_config: Dict[str, Any],
        timeout_override: Optional[int] = None,
        prepare_page: PreparePageCallable = None,
    ) -> Optional[Dict[str, Any]]:
        goto_timeout = timeout_override or int(settings.get("timeout_sec", 60)) * 1000
        if page.url != url:
            await page.goto(url=url, wait_until="domcontentloaded", timeout=goto_timeout)

        if prepare_page:
            try:
                await prepare_page(page)
            except Exception as prep_e:
                self.logger.debug(f"[Extractor] prepare_page skipped: {prep_e}")

        if site.upper() == "MONCLER_OFFICIAL":
            enriched = await self.moncler_extractor.extract(page=page, context=context)
            if enriched:
                return enriched

        price_text = await self._extract_price_with_size_option(page, settings, site_config)
        if price_text:
            data = await extract_title_price(page) or {}
            data["price"] = price_text
            data["url"] = page.url
            return data

        self.logger.warning(f"[PDP] Price not found (even after size attempts) at: {url}")

        ld_json_fallback = await self._extract_ld_json_price(page)
        if ld_json_fallback:
            return ld_json_fallback

        meta_price = await self._extract_meta_price(page)
        if meta_price:
            return meta_price

        return None

    async def _extract_price_with_size_option(self, page: Page, settings: Dict[str, Any], site_config: Optional[Dict[str, Any]] = None) -> Optional[str]:
        price = await self._read_price_or_none(page, site_config)
        if price:
            return price

        policy = settings.get("pdp_size_select_policy", PDPSizeSelectPolicy())
        if policy.mode == "off":
            return None

        self.logger.debug("Price not found initially, attempting size selection...")
        if await self._click_size_to_reveal_price(page, policy, settings, site_config):
            price = await self._read_price_or_none(page, site_config)
            if price:
                self.logger.debug("Price found after size selection.")
                return price
            self.logger.debug("Price NOT found even after size selection.")
        return None

    async def _read_price_or_none(self, page: Page, site_config: Optional[Dict[str, Any]] = None) -> Optional[str]:
        # Stage 3A-2-5: site_config["selectors"]["pdp"]["price"] から取得
        pdp_selectors = (site_config or {}).get("selectors", {}).get("pdp", {}) or {}
        price_selectors = pdp_selectors.get("price", [])
        
        # フォールバック: 空の場合はデフォルトセレクタを使用
        if not price_selectors:
            price_selectors = PRICE_SELECTORS
        
        for selector in price_selectors:
            try:
                locator = page.locator(selector).first
                if await locator.count() == 0:
                    continue
                if selector.startswith("meta["):
                    content = await locator.get_attribute("content")
                else:
                    content = await locator.inner_text()
                price = (content or "").strip()
                if price:
                    return price
            except Exception:
                continue
        return None

    async def _click_size_to_reveal_price(
        self,
        page: Page,
        policy: PDPSizeSelectPolicy,
        settings: Dict[str, Any],
        site_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        # Stage 3A-2-5: site_config["selectors"]["pdp"]["size_button"] から取得
        pdp_selectors = (site_config or {}).get("selectors", {}).get("pdp", {}) or {}
        size_button_selectors = pdp_selectors.get("size_button", [])
        
        # フォールバック: 空の場合はデフォルトセレクタを使用
        if not size_button_selectors:
            size_button_selectors = SIZE_BUTTON_SELECTORS
        
        try:
            buttons = page.locator(", ".join(size_button_selectors))
            count = await buttons.count()
        except Exception:
            return False
        if count == 0:
            return False

        async def btn_label_at(idx: int) -> str:
            try:
                node = buttons.nth(idx)
                aria = await node.get_attribute("aria-label")
                data = await node.get_attribute("data-size")
                txt = await node.inner_text()
                return (data or aria or txt or "").strip()
            except Exception:
                return ""

        clicked = False
        if policy.mode == "by_label" and policy.prefer_labels:
            prefer = [s.strip().upper() for s in policy.prefer_labels if s.strip()]
            for i in range(count):
                try:
                    btn = buttons.nth(i)
                    if not await btn.is_visible() or await btn.is_disabled():
                        continue
                    label = (await btn_label_at(i)).upper()
                    if label and any(label == pref for pref in prefer):
                        await btn.click()
                        clicked = True
                        break
                except Exception:
                    continue

        if not clicked and policy.mode in ("first_instock", "by_label"):
            for i in range(count):
                try:
                    btn = buttons.nth(i)
                    if not await btn.is_visible() or await btn.is_disabled():
                        continue
                    if await btn.get_attribute("aria-disabled") == "true":
                        continue
                    text = (await btn.inner_text() or "").lower()
                    if "out of stock" in text or "在庫なし" in text:
                        continue
                    await btn.click()
                    clicked = True
                    break
                except Exception:
                    continue

        if clicked:
            try:
                await page.wait_for_load_state("networkidle", timeout=2000)
            except Exception:
                self.logger.debug("networkidle timeout after size click.")
            try:
                wait_ms = int(settings.get("pdp_price_wait_ms", 4000))
                # Stage 3A-2-5: site_config から visible_price_selectors を取得
                pdp_selectors = (site_config or {}).get("selectors", {}).get("pdp", {}) or {}
                visible_price_selectors = pdp_selectors.get("visible_price_selectors", [])
                if not visible_price_selectors:
                    visible_price_selectors = VISIBLE_PRICE_SELECTORS
                sel = ", ".join(visible_price_selectors) or "[itemprop=price],[class*=price],[data-testid*=price]"
                await page.wait_for_selector(sel, state="visible", timeout=wait_ms)
            except Exception:
                self.logger.debug("Price selector did not become visible after size click.")
            await page.wait_for_timeout(500)
        return clicked

    async def _extract_ld_json_price(self, page: Page) -> Optional[Dict[str, Any]]:
        try:
            scripts = await page.query_selector_all("script[type='application/ld+json']")
        except Exception:
            return None
        for script in scripts:
            try:
                raw = await script.inner_text()
                data = json.loads(raw or "{}")
            except Exception:
                continue
            if isinstance(data, dict):
                offers = data.get("offers")
                if isinstance(offers, dict):
                    price = offers.get("price")
                    currency = offers.get("priceCurrency")
                    if price:
                        return {"price": str(price), "currency": currency, "url": page.url}
        return None

    async def _extract_meta_price(self, page: Page) -> Optional[Dict[str, Any]]:
        for selector in ("meta[property='og:price:amount']", "meta[name='twitter:data1']"):
            try:
                node = page.locator(selector).first
                if await node.count() == 0:
                    continue
                content = await node.get_attribute("content")
                if content:
                    return {"price": content.strip(), "url": page.url}
            except Exception:
                continue
        return None

    @staticmethod
    def _time_left_ms(start_t: float, budget_ms: int) -> int:
        used = int((time.monotonic() - start_t) * 1000)
        return max(0, budget_ms - used)

    @staticmethod
    def _slice_timeout_ms(left_ms: int, cap_ms: int) -> int:
        return max(500, min(left_ms, cap_ms))

