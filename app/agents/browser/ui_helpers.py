# -*- coding: utf-8 -*-
"""
ui_helpers.py - UI interaction helpers for BrowserUseAgent

This module handles:
- Safe selector waiting
- Overlay/modal dismissal
- Cookie acceptance
- Geo modal handling
- Human-like interactions (mouse, scroll, pause)
- Operator pause for debugging
"""

from __future__ import annotations
import asyncio
import re
from typing import Any, Dict, Optional
from playwright.async_api import Page, Locator

from app.core.run_context import RunContext


def _dedupe_keep_order(items: list) -> list:
    """Remove duplicates while preserving order."""
    return list(dict.fromkeys([i for i in (items or []) if i]))


async def safe_wait_selector(
    page: Page,
    selector: str,
    *,
    timeout_ms: int,
    state: str = "visible"
) -> bool:
    """Safely wait for selector, returning False on failure."""
    if not page or page.is_closed():
        return False
    try:
        await page.wait_for_selector(selector, state=state, timeout=timeout_ms)
        return True
    except Exception:
        return False


async def kill_overlays(page: Page) -> None:
    """Remove overlay elements and unlock body scroll."""
    try:
        await page.evaluate("""
          (() => {
            const sels = ['.overlay','.backdrop','.modal-backdrop','#onetrust-banner-sdk','.cookie-banner','[aria-modal="true"]','.cmp-ui-overlay','.cmp-modal','.drawer--open'];
            document.querySelectorAll(sels.join(',')).forEach(el => el.remove());
            const b = document.body; if (b) { b.classList.remove('modal-open','locked','no-scroll','overflow-hidden'); b.style.overflow=''; }
            const html=document.documentElement; if (html) { html.style.overflow=''; html.classList.remove('no-scroll','overflow-hidden'); }
          })();
        """)
    except Exception:
        pass


async def click_continue_shopping_if_present(
    page: Page,
    site_config: Dict[str, Any]
) -> bool:
    """Click 'Continue Shopping' button if present."""
    ui = (site_config.get("selectors") or {}).get("ui") or {}
    candidates = _dedupe_keep_order(
        (ui.get("continue_shopping") or []) +
        [
            "a:has-text('CONTINUE SHOPPING')",
            "button:has-text('CONTINUE SHOPPING')",
            "[role='button']:has-text('CONTINUE SHOPPING')",
            "text=/\\bCONTINUE\\s+SHOPPING\\b/i"
        ]
    )
    for _ in range(3):
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


async def pause_for_operator(
    page: Optional[Page],
    run_context: Optional[RunContext],
    label: str,
    runtime_kwargs: Dict[str, Any],
    logger: Any,
) -> None:
    """Pause for operator intervention in headful mode."""
    if not runtime_kwargs.get("interactive_pause"):
        return
    slug = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label.lower())
    if page and run_context:
        try:
            await run_context.take_screenshot(page, f"50_operator_{slug}")
        except Exception as e:
            logger.debug(f"[OperatorPause] screenshot failed: {e}")
    prompt = (
        f"\n[OperatorPause] '{label}' で一時停止中です。"
        f" Playwright ウィンドウを操作したら Enter を押して再開してください..."
    )
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: input(prompt))
    except (EOFError, RuntimeError):
        logger.warning("[OperatorPause] input() が使えません。即座に続行します。")


async def accept_cookies_if_present(
    page: Page,
    site_config: Dict[str, Any]
) -> bool:
    """Accept cookies if cookie banner is present."""
    ui = (site_config.get("selectors") or {}).get("ui") or {}
    candidates = _dedupe_keep_order(
        (ui.get("cookie_accept") or []) +
        [
            "#onetrust-accept-btn-handler",
            "button:has-text('ACCEPT ALL')",
            "button:has-text('CONTINUE WITHOUT ACCEPTING')",
            "button[aria-label*='Accept' i]"
        ]
    )
    for sel in candidates:
        try:
            node = page.locator(sel).first
            if await node.count() > 0 and await node.is_visible():
                await node.click(timeout=3000)
                await asyncio.sleep(0.2)
                return True
        except Exception:
            continue
    return False


async def dismiss_geo_modal(page: Page, logger: Any) -> None:
    """
    Dismiss geo/locale modals.
    
    1. Generic "STAY HERE" banners
    2. Moncler "Select your location" locale gate
       - Prefer "UNITED KINGDOM | ENGLISH"
    """
    for sel in [
        "text=STAY HERE",
        "text=REMAIN HERE",
        "text=REMAIN IN ENGLISH",
        "text=CONTINUE SHOPPING",
        "text=ショッピングを続ける",
    ]:
        try:
            el = page.locator(sel).first
            if await el.count() > 0 and await el.is_visible():
                await el.click(timeout=3000)
                break
        except Exception:
            continue

    async def _click_first(loc: Locator, desc: str) -> bool:
        try:
            if await loc.count() == 0:
                return False
            target = loc.first
            if not await target.is_visible():
                await target.scroll_into_view_if_needed()
            await target.click(timeout=5000)
            logger.info(f"[GeoModal] Clicked {desc}")
            await page.wait_for_timeout(300)
            return True
        except Exception as e:
            logger.debug(f"[GeoModal] Click failed ({desc}): {e}")
            return False

    async def _wait_for_en_int(timeout_ms: int = 4000) -> bool:
        try:
            await page.wait_for_function(
                "() => location.href.includes('/en-int/') && !location.href.includes('/en-gb/')",
                timeout=timeout_ms,
            )
            return True
        except Exception:
            return "/en-int/" in (page.url or "").lower()

    try:
        header = page.locator("text=Select your location").first
        header_visible = await header.count() > 0
        if header_visible:
            logger.info("[GeoModal] Moncler locale gate header detected.")

        uk_candidates = [
            page.get_by_text(re.compile(r"UNITED\s+KINGDOM\s*\|\s*ENGLISH", re.I)),
            page.get_by_role("link", name=re.compile(r"UNITED\s+KINGDOM\s*\|\s*ENGLISH", re.I)),
            page.get_by_role("button", name=re.compile(r"UNITED\s+KINGDOM\s*\|\s*ENGLISH", re.I)),
            page.get_by_role("button", name=re.compile(r"United\s+Kingdom.*English", re.I)),
            page.get_by_role("link", name=re.compile(r"United\s+Kingdom.*English", re.I)),
            page.locator("[data-testid*='locale' i] button:has-text('United Kingdom')"),
            page.locator("[data-component*='locale' i] button:has-text('United Kingdom')"),
            page.locator("button:has-text('United Kingdom EN')"),
            page.locator("text=/United\s+Kingdom\s*\|\s*English/i"),
        ]
        for loc in uk_candidates:
            if await _click_first(loc, "United Kingdom / English selector"):
                if await _wait_for_en_int():
                    return
                break

        close_candidates = [
            page.locator("button[aria-label*='close' i]"),
            page.locator("button:has-text('Close')"),
            page.locator(".modal__close, .c-modal__close"),
            page.locator("[data-testid*='close' i]"),
            page.locator("div[data-editorial-component='ticker-top-banner'] button[aria-label*='close' i]"),
        ]
        for loc in close_candidates:
            if await _click_first(loc, "locale gate close button"):
                if await _wait_for_en_int():
                    return

        try:
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(200)
        except Exception:
            pass

        try:
            await page.evaluate(
                """
                () => {
                  const headers = Array.from(
                    document.querySelectorAll('*')
                  ).filter(el => el.textContent && el.textContent.includes('Select your location'));
                  const roots = headers.map(h => h.closest('div[role="dialog"], [data-testid*="modal"], .modal, .c-modal')).filter(Boolean);
                  roots.forEach(root => root.remove());
                }
                """
            )
        except Exception:
            pass
    except Exception:
        return


async def human_like_pause(page: Page, *, min_ms: int = 400, max_ms: int = 900) -> None:
    """Human-like random pause."""
    import random
    await page.wait_for_timeout(random.randint(min_ms, max_ms))


async def human_like_mouse_move(page: Page) -> None:
    """Simulate human-like mouse movements."""
    import random
    try:
        box = await page.evaluate("""() => ({ w: window.innerWidth, h: window.innerHeight })""")
        w, h = int(box.get("w", 1280)), int(box.get("h", 720))
    except Exception:
        w, h = 1280, 720
    moves = random.randint(3, 6)
    for _ in range(moves):
        x = random.randint(int(w * 0.1), int(w * 0.9))
        y = random.randint(int(h * 0.1), int(h * 0.9))
        await page.mouse.move(x, y, steps=random.randint(5, 12))
        await human_like_pause(page, min_ms=120, max_ms=280)


async def human_like_scroll(page: Page) -> None:
    """Simulate human-like scrolling."""
    import random
    try:
        total_height = await page.evaluate("() => document.body ? document.body.scrollHeight : 0")
    except Exception:
        total_height = 0
    if not total_height:
        await page.mouse.wheel(0, random.randint(200, 600))
        await human_like_pause(page, min_ms=200, max_ms=400)
        return
    viewport = await page.evaluate("() => ({h: window.innerHeight || 800})")
    vh = int(viewport.get("h", 800))
    steps = random.randint(2, 4)
    for _ in range(steps):
        delta = random.randint(int(vh * 0.3), int(vh * 0.6))
        await page.mouse.wheel(0, delta)
        await human_like_pause(page, min_ms=200, max_ms=500)

