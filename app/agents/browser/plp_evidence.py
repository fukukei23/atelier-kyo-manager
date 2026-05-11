from __future__ import annotations

import contextlib
import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from playwright.async_api import Page

    from app.core.run_context import RunContext


async def save_materialize_failure_evidence(
    page: Page,
    run_context: RunContext | None,
    tile_selector_str: str,
    plp_config: dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Save evidence of PLP materialize failure (best-effort)."""
    try:
        current_url = page.url

        # 1. DOM snapshot
        try:
            dom_content = await page.content()
            if run_context:
                dom_path = run_context.get_path("plp_failure_dom.html")
                with open(dom_path, "w", encoding="utf-8") as f:
                    f.write(dom_content)
                logger.info(f"[PlpDriver] Saved failure DOM: {dom_path}")
        except Exception as e_dom:
            logger.debug(f"[PlpDriver] Failed to save DOM: {e_dom}")

        # 2. Per-selector tile counts
        try:
            selector_counts: dict[str, int] = {}
            product_tiles = plp_config.get("product_tiles", [])
            product_link = plp_config.get("product_link", [])
            for sel in product_tiles + product_link:
                try:
                    count = await page.locator(sel).count()
                    selector_counts[sel] = count
                except Exception:
                    selector_counts[sel] = -1

            try:
                count = await page.locator(tile_selector_str).count()
                selector_counts["_combined"] = count
            except Exception:
                selector_counts["_combined"] = -1

            if run_context:
                counts_path = run_context.get_path("plp_failure_selector_counts.json")
                with open(counts_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "url": current_url,
                            "selector_counts": selector_counts,
                            "tile_selector_str": tile_selector_str,
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
                logger.info(f"[PlpDriver] Saved selector counts: {counts_path}")
        except Exception as e_counts:
            logger.debug(f"[PlpDriver] Failed to save selector counts: {e_counts}")

        # 3. OneTrust/Consent iframe presence
        try:
            iframe_info: dict[str, dict[str, Any]] = {}
            iframe_selectors = [
                "#onetrust-banner-sdk",
                "#onetrust-pc-sdk",
                "iframe[src*='onetrust']",
                "iframe[src*='consent']",
            ]
            for sel in iframe_selectors:
                try:
                    loc = page.locator(sel)
                    count = await loc.count()
                    is_visible = False
                    if count > 0:
                        with contextlib.suppress(Exception):
                            is_visible = await loc.first.is_visible()
                    iframe_info[sel] = {"count": count, "is_visible": is_visible}
                except Exception:
                    iframe_info[sel] = {"count": -1, "is_visible": False}

            if run_context:
                iframe_path = run_context.get_path("plp_failure_iframe_info.json")
                with open(iframe_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {"url": current_url, "iframe_info": iframe_info},
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
                logger.info(f"[PlpDriver] Saved iframe info: {iframe_path}")
        except Exception as e_iframe:
            logger.debug(f"[PlpDriver] Failed to save iframe info: {e_iframe}")

        # 4. Screenshot
        try:
            if run_context:
                screenshot_path = run_context.get_path("plp_failure_screenshot.png")
                await page.screenshot(path=screenshot_path, full_page=False)
                logger.info(f"[PlpDriver] Saved screenshot: {screenshot_path}")
        except Exception as e_screenshot:
            logger.debug(f"[PlpDriver] Failed to save screenshot: {e_screenshot}")

        # 5. URL info + cookies/localStorage dump
        try:
            if run_context:
                url_info_path = run_context.get_path("plp_failure_url_info.json")
                url_info: dict[str, Any] = {
                    "url": current_url,
                    "title": await page.title() if hasattr(page, "title") else "",
                }
                try:
                    cookies = await page.context.cookies()
                    url_info["cookies_count"] = len(cookies)
                    url_info["cookies_sample"] = cookies[:5] if cookies else []
                except Exception:
                    url_info["cookies_count"] = -1

                try:
                    localStorage_items = await page.evaluate("() => Object.keys(localStorage).length")
                    url_info["localStorage_keys_count"] = localStorage_items
                except Exception:
                    url_info["localStorage_keys_count"] = -1

                with open(url_info_path, "w", encoding="utf-8") as f:
                    json.dump(url_info, f, indent=2, ensure_ascii=False)
                logger.info(f"[PlpDriver] Saved URL info: {url_info_path}")
        except Exception as e_url_info:
            logger.debug(f"[PlpDriver] Failed to save URL info: {e_url_info}")

    except Exception as e:
        logger.debug(f"[PlpDriver] Failed to save materialize failure evidence: {e}")
