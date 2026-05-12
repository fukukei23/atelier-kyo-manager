"""
moncler_nav.py — Moncler固有ナビゲーション Mixin

P1-1 Phase 3: navigation_driver.py から Moncler専用ロジックを抽出。
NavigationDriver に多重継承される Mixin。
"""

from __future__ import annotations

import contextlib
import logging
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from playwright.async_api import Page

from app.agents.browser.nav_types import LinkCandidate, NavigationContext

logger = logging.getLogger(__name__)

from app.agents.browser.extractor import looks_like_product_url


def is_same_origin(url1: str, url2: str) -> bool:
    """2つのURLが同じオリジンか判定"""
    try:
        p1 = urlparse(url1)
        p2 = urlparse(url2)
        return p1.scheme == p2.scheme and p1.netloc == p2.netloc
    except Exception:
        return False


class MonclerNavMixin:
    """Moncler専用ナビゲーションメソッド"""

    page: Page

    async def _collect_moncler_pdp_links(self, ctx: NavigationContext) -> set[str]:
        """CR-ATELIER-002 Step 3: Moncler専用のPDP抽出ロジック"""
        page = self.page
        target_url = page.url
        found_links: set[str] = set()

        moncler_selectors = [
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

        logger.info(f"[PLP→PDP][Moncler] Starting Moncler-specific PDP extraction from URL: {target_url}")

        for sel in moncler_selectors:
            try:
                nodes = await page.query_selector_all(sel)
                if not nodes:
                    continue
                matched_count = 0
                rejected_count = 0
                rejection_reasons = []

                for n in nodes:
                    href = (
                        await n.get_attribute("href")
                        or await n.get_attribute("data-href")
                        or await n.get_attribute("data-product-url")
                        or await n.get_attribute("data-url")
                    )
                    if not href:
                        rejected_count += 1
                        if rejected_count == 1:
                            rejection_reasons.append("no href attribute")
                        continue

                    norm_url = self._normalize_abs_url(target_url, href)

                    if not self._is_valid_moncler_pdp_url(norm_url, target_url):
                        rejected_count += 1
                        if rejected_count == 1:
                            reason = self._get_moncler_rejection_reason(norm_url, target_url)
                            rejection_reasons.append(f"{reason}: {norm_url}")
                        continue

                    found_links.add(norm_url)
                    matched_count += 1

                if matched_count > 0:
                    logger.info(f"[PLP→PDP][Moncler] selector='{sel}' added {matched_count} links")
                elif nodes and rejected_count > 0:
                    logger.debug(
                        f"[PLP→PDP][Moncler] selector='{sel}' found {len(nodes)} elements, "
                        f"but {rejected_count} were rejected. "
                        f"Reasons: {', '.join(rejection_reasons[:2])}"
                    )
            except Exception as e:
                logger.debug(f"[PLP→PDP][Moncler] selector='{sel}' failed: {e}")

        # Phase 2: グローバルsweep
        if not found_links:
            logger.debug("[PLP→PDP][Moncler] Selector-based extraction found no links, trying global sweep...")
            try:
                raw_hrefs: list[str] = await page.evaluate(
                    "() => Array.from(document.querySelectorAll('a[href]')).map(a => a.getAttribute('href')).filter(Boolean)"
                )
                pdp_rx = re.compile(r"/products?/", re.I)
                for href in raw_hrefs:
                    if "onetrust.com" in href.lower():
                        continue
                    if pdp_rx.search(href):
                        norm_url = self._normalize_abs_url(target_url, href)
                        if self._is_valid_moncler_pdp_url(norm_url, target_url):
                            found_links.add(norm_url)
                if found_links:
                    logger.info(f"[PLP→PDP][Moncler] Global sweep found {len(found_links)} links")
            except Exception as e:
                logger.warning(f"[PLP→PDP][Moncler] Global sweep failed: {e}")

        if found_links:
            sample_urls = list(found_links)[:5]
            logger.debug(f"[PLP→PDP][Moncler] Sample PDP URLs: {sample_urls}")
        else:
            logger.warning("[PLP→PDP][Moncler] No valid PDP links found after Moncler-specific extraction")

        return found_links

    def _is_valid_moncler_pdp_url(self, url: str, base_url: str) -> bool:
        """CR-ATELIER-002 Step 3: Moncler専用のURLバリデーション"""
        try:
            parsed = urlparse(url)

            if parsed.scheme not in ("http", "https"):
                return False

            host = parsed.netloc.lower()
            if not host.endswith("moncler.com"):
                return False

            blocked_domains = [
                "onetrust.com",
                "monclergroup.com",
                "facebook.com",
                "twitter.com",
                "instagram.com",
                "pinterest.com",
            ]
            for blocked in blocked_domains:
                if blocked in host:
                    return False

            path = parsed.path or ""
            if not re.search(r"/products?/", path, re.I):
                return False

            trap_patterns = [
                r"/404",
                r"/not-found",
                r"/search\?",
                r"/legal/",
                r"/client-service",
            ]
            for pattern in trap_patterns:
                if re.search(pattern, path, re.I):
                    return False

            if not is_same_origin(url, base_url):
                return False

            return looks_like_product_url(url)
        except Exception as e:
            logger.debug(f"[PLP→PDP][Moncler] URL validation failed for {url}: {e}")
            return False

    def _get_moncler_rejection_reason(self, url: str, base_url: str) -> str:
        """Moncler URLバリデーションでrejectされた理由を取得"""
        try:
            parsed = urlparse(url)
            host = parsed.netloc.lower()
            path = parsed.path or ""

            if not host.endswith("moncler.com"):
                return "external_domain"
            if "onetrust.com" in host:
                return "blocked_domain_onetrust"
            if "monclergroup.com" in host:
                return "blocked_domain_monclergroup"
            if not re.search(r"/products?/", path, re.I):
                return "no_products_path"
            if not is_same_origin(url, base_url):
                return "different_origin"
            if not looks_like_product_url(url):
                return "not_product_url_pattern"

            return "unknown"
        except Exception:
            return "validation_error"

    async def _trigger_moncler_self_healing(
        self,
        ctx: NavigationContext,
        failure_reason: str,
        outcome_dict: dict[str, Any],
    ) -> None:
        """CR-ATELIER-002 Step 6-3: Moncler 専用の Self-Healing / Selector Discovery"""
        try:
            from app.agents.moncler_patch_builder import process_moncler_self_healing_results
            from app.agents.selector_discovery_agent import SelectorDiscoveryAgent
            from app.agents.healing.self_healing_agent import SelfHealingAgent

            dom_snapshot_path = None
            if ctx.run_context:
                with contextlib.suppress(Exception):
                    dom_snapshot_path = str(ctx.run_context.get_path("failure_dom.html"))

            failure_payload = {
                "site": "MONCLER_OFFICIAL",
                "url": self.page.url or ctx.entry_url or "",
                "failure_reason": failure_reason,
                "dom_snapshot_path": dom_snapshot_path,
                "layer_stats": outcome_dict.get("layer_stats", {}),
                "rejection_stats": {},
                "selectors_current": ctx.site_config.get("selectors", {}),
                "run_id": getattr(ctx.run_context, "run_id", None) if ctx.run_context else None,
                "timestamp": outcome_dict.get("timestamp"),
            }

            self_healing_result = None
            try:
                self_healing_agent = SelfHealingAgent()
                if hasattr(self_healing_agent, "handle_moncler_failure"):
                    self_healing_result = await self_healing_agent.handle_moncler_failure(failure_payload)
                    logger.info(
                        f"[SelfHealing][Moncler] Self-healing result: {self_healing_result.get('analysis', 'N/A')}"
                    )
                else:
                    logger.debug("[SelfHealing][Moncler] handle_moncler_failure not implemented, skipping")
            except Exception as e:
                logger.warning(f"[SelfHealing][Moncler] Failed to call self-healing agent: {e}", exc_info=True)

            selector_discovery_result = None
            try:
                selector_discovery_agent = SelectorDiscoveryAgent()
                if hasattr(selector_discovery_agent, "propose_moncler_selectors"):
                    discovery_payload = {
                        "dom_snapshot_path": dom_snapshot_path,
                        "selectors_current": ctx.site_config.get("selectors", {}),
                        "layer_stats": outcome_dict.get("layer_stats", {}),
                        "rejection_stats": {},
                        "run_id": getattr(ctx.run_context, "run_id", None) if ctx.run_context else None,
                    }
                    selector_discovery_result = await selector_discovery_agent.propose_moncler_selectors(
                        discovery_payload
                    )
                    logger.info(
                        f"[SelectorDiscovery][Moncler] Proposed {len(selector_discovery_result.get('candidate_selectors', []))} selectors"
                    )
                else:
                    logger.debug("[SelectorDiscovery][Moncler] propose_moncler_selectors not implemented, skipping")
            except Exception as e:
                logger.warning(
                    f"[SelectorDiscovery][Moncler] Failed to call selector discovery agent: {e}", exc_info=True
                )

            if ctx.run_context and (self_healing_result or selector_discovery_result):
                try:
                    run_id = getattr(ctx.run_context, "run_id", None) or "unknown"
                    site = ctx.site_config.get("site_code") or ctx.site_config.get("site") or "MONCLER_OFFICIAL"
                    current_url = self.page.url or ctx.entry_url or ""

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

                    saved_paths = await process_moncler_self_healing_results(
                        run_context=ctx.run_context,
                        run_id=run_id,
                        site=site,
                        current_url=current_url,
                        moncler_outcome=moncler_outcome,
                        self_healing_result=self_healing_result,
                        selector_discovery_result=selector_discovery_result,
                        current_site_config=ctx.site_config,
                        generate_markdown=True,
                    )

                    if saved_paths:
                        logger.info(
                            f"[PatchBuilder][Moncler] Generated patch files: "
                            f"analysis={saved_paths.get('analysis')}, "
                            f"patch={saved_paths.get('patch_candidate')}"
                        )
                except Exception as e:
                    logger.warning(f"[PatchBuilder][Moncler] Failed to generate patch files: {e}", exc_info=True)
        except Exception as e:
            logger.warning(
                f"[SelfHealing][Moncler] Failed to trigger self-healing/selector discovery: {e}", exc_info=True
            )
