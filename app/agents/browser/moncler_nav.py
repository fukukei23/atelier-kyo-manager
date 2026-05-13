"""Moncler固有ナビゲーション Mixin — NavigationDriver に多重継承される。"""
from __future__ import annotations

import contextlib
import logging
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from playwright.async_api import Page

from app.agents.browser.nav_types import NavigationContext
from app.agents.browser.extractor import looks_like_product_url

logger = logging.getLogger(__name__)

_MONCLER_SELECTORS = [
    "article[data-component*='ProductCard'] a[href*='/products/']",
    "article[data-component*='ProductCard'] a[href*='/product/']",
    "[data-component*='ProductCard'] a[href*='/products/']",
    "[data-component*='ProductCard'] a[href*='/product/']",
    "[data-testid*='product-card'] a[href*='/products/']",
    "[data-testid*='product-tile'] a[href*='/products/']",
    "[data-test*='product-card'] a[href*='/products/']",
    ".product-card a[href*='/products/']",
    ".c-product-card a[href*='/products/']",
    ".product-tile a[href*='/products/']",
    ".c-product-tile a[href*='/products/']",
    "a[href*='/en-int/products/']",
    "a[href*='/products/']",
    "[data-qa='product-tile'] a[href*='/products/']",
    "[data-qa*='product'] a[href*='/products/']",
]

_BLOCKED_DOMAINS = ("onetrust.com", "monclergroup.com", "facebook.com", "twitter.com", "instagram.com", "pinterest.com")
_TRAP_PATTERNS = (r"/404", r"/not-found", r"/search\?", r"/legal/", r"/client-service")
_PDP_RX = re.compile(r"/products?/", re.I)
_TRAP_RX = re.compile("|".join(_TRAP_PATTERNS), re.I)


def is_same_origin(url1: str, url2: str) -> bool:
    try:
        p1, p2 = urlparse(url1), urlparse(url2)
        return p1.scheme == p2.scheme and p1.netloc == p2.netloc
    except Exception:
        return False


async def _extract_href(node) -> str | None:
    return (
        await node.get_attribute("href")
        or await node.get_attribute("data-href")
        or await node.get_attribute("data-product-url")
        or await node.get_attribute("data-url")
    )


class MonclerNavMixin:

    page: Page

    # ── PDP collection ──────────────────────────────────

    async def _collect_moncler_pdp_links(self, ctx: NavigationContext) -> set[str]:
        page = self.page
        target_url = page.url
        found: set[str] = set()

        logger.info(f"[PLP→PDP][Moncler] Starting from: {target_url}")

        for sel in _MONCLER_SELECTORS:
            matched, rejected = await self._try_selector(page, sel, target_url, found)
            if matched > 0:
                logger.info(f"[PLP→PDP][Moncler] selector='{sel}' added {matched} links")
            elif rejected > 0:
                logger.debug(f"[PLP→PDP][Moncler] selector='{sel}' rejected {rejected}")

        if not found:
            await self._global_sweep(page, target_url, found)

        if found:
            logger.debug(f"[PLP→PDP][Moncler] Sample URLs: {list(found)[:5]}")
        else:
            logger.warning("[PLP→PDP][Moncler] No valid PDP links found")

        return found

    async def _try_selector(self, page, selector: str, target_url: str, found: set[str]) -> tuple[int, int]:
        matched, rejected = 0, 0
        try:
            nodes = await page.query_selector_all(selector)
            if not nodes:
                return 0, 0
            for n in nodes:
                href = await _extract_href(n)
                if not href:
                    rejected += 1
                    continue
                norm_url = self._normalize_abs_url(target_url, href)
                if self._is_valid_moncler_pdp_url(norm_url, target_url):
                    found.add(norm_url)
                    matched += 1
                else:
                    rejected += 1
        except Exception as e:
            logger.debug(f"[PLP→PDP][Moncler] selector='{selector}' failed: {e}")
        return matched, rejected

    async def _global_sweep(self, page, target_url: str, found: set[str]) -> None:
        logger.debug("[PLP→PDP][Moncler] Trying global sweep...")
        try:
            raw_hrefs: list[str] = await page.evaluate(
                "() => Array.from(document.querySelectorAll('a[href]')).map(a => a.getAttribute('href')).filter(Boolean)"
            )
            for href in raw_hrefs:
                if "onetrust.com" in href.lower():
                    continue
                if _PDP_RX.search(href):
                    norm_url = self._normalize_abs_url(target_url, href)
                    if self._is_valid_moncler_pdp_url(norm_url, target_url):
                        found.add(norm_url)
            if found:
                logger.info(f"[PLP→PDP][Moncler] Global sweep found {len(found)} links")
        except Exception as e:
            logger.warning(f"[PLP→PDP][Moncler] Global sweep failed: {e}")

    # ── URL validation ──────────────────────────────────

    def _is_valid_moncler_pdp_url(self, url: str, base_url: str) -> bool:
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return False
            host = parsed.netloc.lower()
            if not host.endswith("moncler.com"):
                return False
            if any(d in host for d in _BLOCKED_DOMAINS):
                return False
            path = parsed.path or ""
            if not _PDP_RX.search(path):
                return False
            if _TRAP_RX.search(path):
                return False
            if not is_same_origin(url, base_url):
                return False
            return looks_like_product_url(url)
        except Exception as e:
            logger.debug(f"[PLP→PDP][Moncler] URL validation failed for {url}: {e}")
            return False

    def _get_moncler_rejection_reason(self, url: str, base_url: str) -> str:
        try:
            parsed = urlparse(url)
            host = parsed.netloc.lower()
            path = parsed.path or ""
            if not host.endswith("moncler.com"):
                return "external_domain"
            for d in _BLOCKED_DOMAINS:
                if d in host:
                    return f"blocked_domain_{d.split('.')[0]}"
            if not _PDP_RX.search(path):
                return "no_products_path"
            if not is_same_origin(url, base_url):
                return "different_origin"
            if not looks_like_product_url(url):
                return "not_product_url_pattern"
            return "unknown"
        except Exception:
            return "validation_error"

    # ── Self-healing ────────────────────────────────────

    async def _trigger_moncler_self_healing(
        self, ctx: NavigationContext, failure_reason: str, outcome_dict: dict[str, Any],
    ) -> None:
        try:
            from app.agents.moncler_patch_builder import process_moncler_self_healing_results
            from app.agents.selector_discovery_agent import SelectorDiscoveryAgent
            from app.agents.healing.self_healing_agent import SelfHealingAgent

            dom_path = None
            if ctx.run_context:
                with contextlib.suppress(Exception):
                    dom_path = str(ctx.run_context.get_path("failure_dom.html"))

            run_id = getattr(ctx.run_context, "run_id", None) if ctx.run_context else None
            selectors_current = ctx.site_config.get("selectors", {})

            self_healing_result = await self._run_self_healing(SelfHealingAgent, failure_reason, dom_path, outcome_dict, selectors_current, run_id)
            discovery_result = await self._run_selector_discovery(SelectorDiscoveryAgent, dom_path, selectors_current, outcome_dict, run_id)

            if ctx.run_context and (self_healing_result or discovery_result):
                await self._save_healing_results(ctx, outcome_dict, self_healing_result, discovery_result)

        except Exception as e:
            logger.warning(f"[SelfHealing][Moncler] Failed: {e}", exc_info=True)

    async def _run_self_healing(self, agent_cls, failure_reason: str, dom_path, outcome_dict, selectors, run_id) -> dict | None:
        try:
            agent = agent_cls()
            if not hasattr(agent, "handle_moncler_failure"):
                logger.debug("[SelfHealing][Moncler] handle_moncler_failure not implemented")
                return None
            payload = {
                "site": "MONCLER_OFFICIAL",
                "url": self.page.url or "",
                "failure_reason": failure_reason,
                "dom_snapshot_path": dom_path,
                "layer_stats": outcome_dict.get("layer_stats", {}),
                "rejection_stats": {},
                "selectors_current": selectors,
                "run_id": run_id,
                "timestamp": outcome_dict.get("timestamp"),
            }
            result = await agent.handle_moncler_failure(payload)
            logger.info(f"[SelfHealing][Moncler] {result.get('analysis', 'N/A')}")
            return result
        except Exception as e:
            logger.warning(f"[SelfHealing][Moncler] Failed: {e}", exc_info=True)
            return None

    async def _run_selector_discovery(self, agent_cls, dom_path, selectors, outcome_dict, run_id) -> dict | None:
        try:
            agent = agent_cls()
            if not hasattr(agent, "propose_moncler_selectors"):
                logger.debug("[SelectorDiscovery][Moncler] propose_moncler_selectors not implemented")
                return None
            payload = {
                "dom_snapshot_path": dom_path,
                "selectors_current": selectors,
                "layer_stats": outcome_dict.get("layer_stats", {}),
                "rejection_stats": {},
                "run_id": run_id,
            }
            result = await agent.propose_moncler_selectors(payload)
            n = len(result.get("candidate_selectors", []))
            logger.info(f"[SelectorDiscovery][Moncler] Proposed {n} selectors")
            return result
        except Exception as e:
            logger.warning(f"[SelectorDiscovery][Moncler] Failed: {e}", exc_info=True)
            return None

    async def _save_healing_results(self, ctx, outcome_dict, self_healing_result, discovery_result) -> None:
        from app.agents.moncler_patch_builder import process_moncler_self_healing_results

        try:
            run_id = getattr(ctx.run_context, "run_id", None) or "unknown"
            site = ctx.site_config.get("site_code") or ctx.site_config.get("site") or "MONCLER_OFFICIAL"
            moncler_outcome = {
                "plp_materialized": outcome_dict.get("plp_materialized", False),
                "tiles_detected": outcome_dict.get("tiles_detected", 0),
                "pdp_links_raw": outcome_dict.get("pdp_links_raw", 0),
                "pdp_links_accepted": outcome_dict.get("pdp_links_accepted", 0),
                "layer_stats": outcome_dict.get("layer_stats", {}),
                "rejection_stats": {},
                "locale_corrections": outcome_dict.get("locale_corrections", 0),
                "trap_detected": outcome_dict.get("trap_detected", False),
            }
            saved = await process_moncler_self_healing_results(
                run_context=ctx.run_context, run_id=run_id, site=site,
                current_url=self.page.url or ctx.entry_url or "",
                moncler_outcome=moncler_outcome,
                self_healing_result=self_healing_result,
                selector_discovery_result=discovery_result,
                current_site_config=ctx.site_config, generate_markdown=True,
            )
            if saved:
                logger.info(f"[PatchBuilder][Moncler] analysis={saved.get('analysis')} patch={saved.get('patch_candidate')}")
        except Exception as e:
            logger.warning(f"[PatchBuilder][Moncler] Failed: {e}", exc_info=True)
