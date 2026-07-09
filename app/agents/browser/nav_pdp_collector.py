from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, TypedDict

from playwright.async_api import Page

from app.agents.browser.nav_types import (
    LinkCandidate,
    NavigationContext,
    RejectReason,
)
from app.agents.browser.url_rules import (
    classify_candidate,
    is_same_site,
    normalize_candidate_url,
)

logger: logging.Logger = logging.getLogger(__name__)


# ── Types ────────────────────────────────────────────────


class ValidationStats(TypedDict, total=False):
    reject_reason_counts: dict[str, int]
    top_reject_reasons: dict[str, int]
    domain_rejected_count: int
    path_rejected_count: int
    allow_path_matched_count: int
    forbidden_path_matched_count: int
    allow_path_pattern_counts: dict[str, int]
    forbidden_path_pattern_counts: dict[str, int]


class LinkEvidence(TypedDict, total=False):
    total_candidates: int
    total_valid: int
    top_reject_reasons: dict[str, int]
    sample_candidates: list[dict[str, Any]]
    error: str


# ── Helpers ─────────────────────────────────────────────


def _dedupe_keep_order(items: list[str]) -> list[str]:
    return list(dict.fromkeys([i for i in (items or []) if i]))


async def _extract_url_from_node(node: Any) -> str | None:
    return (
        await node.get_attribute("href")
        or await node.get_attribute("data-href")
        or await node.get_attribute("data-product-url")
        or await node.get_attribute("data-url")
    )


def _candidate_to_dict(candidate: LinkCandidate) -> dict[str, Any]:
    return {
        "phase": candidate.phase,
        "source_selector": candidate.source_selector or "",
        "raw_href": candidate.url,
        "resolved_url": candidate.normalized_url or "",
        "origin": candidate.origin or "",
        "passed": candidate.accepted,
        "reject_reasons": candidate.reject_reasons if candidate.reject_reasons else [],
        "notes": candidate.notes or "",
        "product_url_rules": candidate.product_url_rules if candidate.product_url_rules else {},
    }


# ── Constants ────────────────────────────────────────────

_DEFAULT_PDP_SELECTORS: tuple[str, ...] = (
    "a[href*='/products/']",
    "a[href*='/product/']",
    "a[href*='/p/']",
    "[data-component*='ProductCard'] a[href]",
    "[class*='product-card'] a[href]",
    "article [data-testid*='product']:is(a, * a)",
    "[data-testid*='card'] a[href]",
    "[data-testid*='product-card'] a[href]",
    "a[data-product-url]",
    "[data-qa='product-tile'] a[href]",
)

_NOISE_RX: re.Pattern[str] = re.compile(
    r"/(collections?|seasons?|client-service|login|legal|cart|wishlist|search)/", re.IGNORECASE
)
_PDP_RX: re.Pattern[str] = re.compile(r"/(products?|p)/", re.IGNORECASE)
_HTML_PDP_RX: re.Pattern[str] = re.compile(r"/(?:en-int|en-jp/en-int)/[^/]+/[^/]+/[^/]+\.html", re.IGNORECASE)


class NavPdpCollectorMixin:
    def _create_candidate(
        self,
        href: str,
        phase: str,
        source_selector: str,
        target_url: str,
        site_config: dict[str, Any],
        all_candidates: list[LinkCandidate],
        found_links: set[str],
    ) -> LinkCandidate:
        norm_url, norm_info = normalize_candidate_url(href, target_url, site_config)
        candidate = LinkCandidate(url=href, phase=phase, normalized_url=norm_url, source_selector=source_selector)
        candidate = classify_candidate(candidate, target_url, site_config)
        if candidate.product_url_rules:
            candidate.product_url_rules["normalization_info"] = norm_info
        all_candidates.append(candidate)
        if candidate.accepted:
            found_links.add(norm_url)
        return candidate

    async def _phase1a_global_sweep(
        self,
        page: Page,
        target_url: str,
        site_config: dict[str, Any],
        all_candidates: list[LinkCandidate],
        found_links: set[str],
    ) -> None:
        try:
            raw_hrefs: list[str] = await page.evaluate(
                "() => Array.from(document.querySelectorAll('a[href]')).map(a => a.getAttribute('href')).filter(Boolean)"
            )
        except Exception as e:
            logger.warning("[PLP→PDP][1a] Sweep failed: %s", e)
            raw_hrefs = []

        for href in raw_hrefs:
            if "onetrust.com" in href.lower():
                continue
            if _PDP_RX.search(href) or _HTML_PDP_RX.search(href):
                self._create_candidate(href, "1a", "global_sweep", target_url, site_config, all_candidates, found_links)

        if found_links:
            logger.info("[PLP→PDP][1a] Sweep found %d links.", len(found_links))

    async def _phase1b_selector_based(
        self,
        page: Page,
        target_url: str,
        site_config: dict[str, Any],
        all_candidates: list[LinkCandidate],
        found_links: set[str],
    ) -> list[str]:
        plp = (site_config.get("selectors", {}) or {}).get("plp", {}) or {}
        pdp = (site_config.get("selectors", {}) or {}).get("pdp", {}) or {}
        selectors = _dedupe_keep_order(
            (plp.get("pdp_link_selectors", []) or []) + (pdp.get("pdp_link_selectors", []) or [])
        ) or list(_DEFAULT_PDP_SELECTORS)

        for sel in selectors:
            try:
                nodes = await page.query_selector_all(sel)
                if not nodes:
                    continue
                matched, rejected = 0, 0
                for n in nodes:
                    href = await _extract_url_from_node(n)
                    if not href:
                        all_candidates.append(
                            LinkCandidate(
                                url="",
                                phase="1b",
                                normalized_url="",
                                reject_reasons=[RejectReason.NO_HREF.value],
                                source_selector=sel,
                            )
                        )
                        rejected += 1
                        continue
                    if "onetrust.com" in href.lower():
                        continue
                    c = self._create_candidate(href, "1b", sel, target_url, site_config, all_candidates, found_links)
                    if c.accepted:
                        matched += 1
                    else:
                        rejected += 1
                if matched > 0:
                    logger.info("[PLP→PDP][1b] selector='%s' added %d links.", sel, matched)
                elif nodes and rejected > 0:
                    logger.warning(
                        "[PLP→PDP][1b] selector='%s' found %d elements, but %d were rejected.",
                        sel,
                        len(nodes),
                        rejected,
                    )
            except Exception as e:
                logger.warning("[PLP→PDP][1b] selector='%s' failed: %s", sel, e)
        return selectors

    async def _phase2_deep_extraction(
        self,
        page: Page,
        target_url: str,
        site_config: dict[str, Any],
        all_candidates: list[LinkCandidate],
        found_links: set[str],
    ) -> None:
        if found_links:
            return
        logger.warning("[PLP→PDP] Phase 1a/1b found no links. Falling back to Phase 2 (Deep Extraction)...")
        try:
            deep_hrefs = await self._run_deep_extraction_phase2(page, site_config)  # type: ignore[attr-defined]
            for href in deep_hrefs:
                self._create_candidate(
                    href, "2", "deep_extraction", target_url, site_config, all_candidates, found_links
                )
            if found_links:
                logger.info("[PLP→PDP][2] Deep Extraction found %d links.", len(found_links))
        except Exception as e:
            logger.error("[PLP→PDP][2] Deep Extraction failed: %s", e)

    def _refilter_candidates(
        self,
        all_candidates: list[LinkCandidate],
        found_links: set[str],
        target_url: str,
        site_config: dict[str, Any],
    ) -> None:
        if found_links or not all_candidates:
            return
        refilter_config = (site_config or {}).get("pdp_link_refilter", {})
        if not refilter_config.get("enabled", False):
            return

        logger.info(
            "[PLP→PDP][Refilter] %d candidates found but all rejected. Attempting evidence-based relaxed filtering...",
            len(all_candidates),
        )

        product_card_keywords = ("product", "card", "tile", "item")

        for candidate in all_candidates:
            if not candidate.normalized_url:
                continue
            if (
                candidate.product_url_rules
                and not candidate.product_url_rules.get("forbidden_path_matched", True)
                and candidate.product_url_rules.get("domain_allowed", False)
                and candidate.source_selector
                and any(kw in candidate.source_selector.lower() for kw in product_card_keywords)
            ):
                found_links.add(candidate.normalized_url)
                candidate.accepted = True
                candidate.reject_reasons = [
                    r for r in candidate.reject_reasons if r not in (RejectReason.NO_PRODUCTS_PATH.value,)
                ]
                candidate.notes = (candidate.notes or "") + " [Refilter: product card source]"

        if found_links:
            logger.info("[PLP→PDP][Refilter] Evidence-based relaxed filtering accepted %d links.", len(found_links))
            return

        if refilter_config.get("allow_same_site_only", False):
            for candidate in all_candidates:
                if not candidate.normalized_url:
                    continue
                if (
                    is_same_site(candidate.normalized_url, target_url)
                    and candidate.product_url_rules
                    and not candidate.product_url_rules.get("forbidden_path_matched", False)
                ):
                    relaxed_reasons = [
                        r
                        for r in candidate.reject_reasons
                        if r
                        not in (
                            RejectReason.DIFFERENT_DOMAIN.value,
                            RejectReason.DIFFERENT_SUBDOMAIN.value,
                            RejectReason.DIFFERENT_ORIGIN.value,
                        )
                    ]
                    if not relaxed_reasons:
                        found_links.add(candidate.normalized_url)
                        candidate.accepted = True
                        candidate.reject_reasons = []

        if found_links:
            return

        ignore_reasons = refilter_config.get("ignore_reject_reasons", [])
        if not ignore_reasons:
            return
        for candidate in all_candidates:
            if not candidate.normalized_url:
                continue
            if candidate.product_url_rules and not candidate.product_url_rules.get("forbidden_path_matched", False):
                filtered_reasons = [r for r in candidate.reject_reasons if r not in ignore_reasons]
                if not filtered_reasons:
                    found_links.add(candidate.normalized_url)
                    candidate.accepted = True
                    candidate.reject_reasons = []
                    candidate.notes = (candidate.notes or "") + f" [Refilter: ignored {ignore_reasons}]"

    async def _diagnose_no_links(
        self,
        page: Page,
        target_url: str,
        selectors: list[str],
        ctx: NavigationContext,
    ) -> None:
        logger.error(
            "[PLP→PDP] No PDP hrefs found after all phases. Target URL: %s. "
            "This indicates a selector mismatch or URL validation failure.",
            target_url,
        )
        try:
            total = 0
            for sel in selectors:
                try:
                    nodes = await page.query_selector_all(sel)
                    if nodes:
                        total += len(nodes)
                        if total == len(nodes):
                            sample = nodes[0]
                            attrs = {
                                "tag": await sample.evaluate("el => el.tagName"),
                                "href": await sample.get_attribute("href"),
                                "data-href": await sample.get_attribute("data-href"),
                                "data-product-url": await sample.get_attribute("data-product-url"),
                                "class": await sample.get_attribute("class"),
                            }
                            logger.debug("[PLP→PDP] Sample element from selector '%s': %s", sel, attrs)
                except Exception:
                    pass
            if total > 0:
                logger.warning("[PLP→PDP] Found %d elements matching selectors, but none passed validation.", total)
        except Exception as e:
            logger.debug("[PLP→PDP] Diagnostic info collection failed: %s", e)

    async def collect_pdp_links(self, ctx: NavigationContext) -> list[str]:
        page: Page = self.page  # type: ignore[attr-defined]
        site_config = ctx.site_config
        run_context = ctx.run_context
        target_url = page.url
        found_links: set[str] = set()
        all_candidates: list[LinkCandidate] = []

        await self._phase1a_global_sweep(page, target_url, site_config, all_candidates, found_links)
        selectors = await self._phase1b_selector_based(page, target_url, site_config, all_candidates, found_links)
        await self._phase2_deep_extraction(page, target_url, site_config, all_candidates, found_links)
        self._refilter_candidates(all_candidates, found_links, target_url, site_config)

        links = sorted(found_links)
        if not links:
            await self._diagnose_no_links(page, target_url, selectors, ctx)

        cleaned = [u for u in links if not _NOISE_RX.search(u)]
        logger.info("[PLP→PDP] collected %d PDP-like links (raw=%d)", len(cleaned), len(links))

        link_collection_summary = await self._save_link_collection_evidence(
            all_candidates=all_candidates,
            accepted_links=cleaned,
            run_context=run_context,
            ctx=ctx,
        )
        ctx.link_collection_summary = link_collection_summary

        try:
            sample = cleaned[:20]
            if self.telemetry and ctx.run_context:  # type: ignore[attr-defined]
                from app.agents.browser.telemetry import TelemetryContext

                tctx = TelemetryContext(
                    site=ctx.site, query=ctx.query, run_id=getattr(ctx.run_context, "run_id", None), stage="plp"
                )
                await self.telemetry.save_json("raw_pdp_links_v85.5", {"links": cleaned, "sample": sample}, tctx)  # type: ignore[attr-defined]
                await self.telemetry._service.save_raw_hrefs(cleaned, name="raw_hrefs_final_cleaned")  # type: ignore[attr-defined]
            elif ctx.run_context and hasattr(ctx.run_context, "save_json"):
                ctx.run_context.save_json("raw_pdp_links_v85.5.json", {"links": cleaned, "sample": sample})
                from app.utils.observability import save_raw_hrefs

                if callable(save_raw_hrefs):
                    res = save_raw_hrefs(ctx.run_context, cleaned, name="raw_hrefs_final_cleaned")
                    if asyncio.iscoroutine(res):
                        await res
        except Exception as e:
            logger.debug("[PLP→PDP] TelemetryClient.save_raw_hrefs failed: %s", e)

        return cleaned

    def _compute_validation_stats(self, all_candidates: list[LinkCandidate]) -> ValidationStats:
        reject_reason_counts: dict[str, int] = {}
        domain_rejected = 0
        path_rejected = 0
        allow_matched = 0
        forbidden_matched = 0
        allow_patterns: dict[str, int] = {}
        forbidden_patterns: dict[str, int] = {}

        for candidate in all_candidates:
            for reason in candidate.reject_reasons:
                reject_reason_counts[reason] = reject_reason_counts.get(reason, 0) + 1
            if candidate.product_url_rules:
                rules = candidate.product_url_rules
                if not rules.get("domain_allowed", True):
                    domain_rejected += 1
                if rules.get("forbidden_path_matched", False):
                    forbidden_matched += 1
                    p = rules.get("forbidden_path_pattern")
                    if p:
                        forbidden_patterns[p] = forbidden_patterns.get(p, 0) + 1
                if rules.get("allow_path_matched", False):
                    allow_matched += 1
                    p = rules.get("allow_path_pattern")
                    if p:
                        allow_patterns[p] = allow_patterns.get(p, 0) + 1
                if not rules.get("allow_path_matched", False) and not rules.get("forbidden_path_matched", False):
                    path_rejected += 1

        return {
            "reject_reason_counts": reject_reason_counts,
            "top_reject_reasons": dict(sorted(reject_reason_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            "domain_rejected_count": domain_rejected,
            "path_rejected_count": path_rejected,
            "allow_path_matched_count": allow_matched,
            "forbidden_path_matched_count": forbidden_matched,
            "allow_path_pattern_counts": dict(sorted(allow_patterns.items(), key=lambda x: x[1], reverse=True)[:10]),
            "forbidden_path_pattern_counts": dict(
                sorted(forbidden_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
        }

    def _save_report_safely(self, run_context: Any, report: dict[str, Any]) -> None:
        try:
            run_context.save_json("pdp_link_validation_report.json", report)
        except OSError as e:
            logger.error("[PLP→PDP] Failed to save validation report: %s", e, exc_info=True)
            try:
                path = run_context.get_path("pdp_link_validation_report.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
            except OSError as e2:
                logger.error("[PLP→PDP] Failed to write validation report directly: %s", e2, exc_info=True)

    async def _save_link_collection_evidence(
        self,
        all_candidates: list[LinkCandidate],
        accepted_links: list[str],
        run_context: Any | None,
        ctx: NavigationContext,
    ) -> LinkEvidence:
        summary: LinkEvidence = {
            "total_candidates": len(all_candidates),
            "total_valid": len(accepted_links),
            "top_reject_reasons": {},
            "sample_candidates": [],
        }
        if not run_context or not hasattr(run_context, "save_json"):
            return summary

        try:
            phase1: list[dict[str, Any]] = []
            phase2: list[dict[str, Any]] = []
            phase3: list[dict[str, Any]] = []

            for candidate in all_candidates[:200]:
                d = _candidate_to_dict(candidate)
                if candidate.phase in ("1a", "1b"):
                    phase1.append(d)
                elif candidate.phase == "2":
                    phase2.append(d)
                elif candidate.phase == "3":
                    phase3.append(d)

            run_context.save_json(
                "pdp_link_candidates_phase1.json", {"phase": "1a/1b", "candidates": phase1, "total": len(phase1)}
            )
            run_context.save_json(
                "pdp_link_candidates_phase2.json", {"phase": "2", "candidates": phase2, "total": len(phase2)}
            )
            if phase3:
                run_context.save_json(
                    "pdp_link_candidates_phase3.json", {"phase": "3", "candidates": phase3, "total": len(phase3)}
                )

            stats = self._compute_validation_stats(all_candidates)

            rejected = [c for c in all_candidates if not c.accepted]
            validation_report: dict[str, Any] = {
                "total_candidates": len(all_candidates),
                "total_valid": len(accepted_links),
                "total_rejected": len(rejected),
                **stats,
                "all_rejected_candidates": [_candidate_to_dict(c) for c in rejected[:200]],
                "sample_rejected": [_candidate_to_dict(c) for c in rejected[:10]],
            }
            self._save_report_safely(run_context, validation_report)

            summary["total_candidates"] = len(all_candidates)
            summary["total_valid"] = len(accepted_links)
            summary["top_reject_reasons"] = stats["top_reject_reasons"]
            summary["sample_candidates"] = [_candidate_to_dict(c) for c in all_candidates[:10]]

            logger.info(
                "[PLP→PDP][CR-E2E-003A] Collected %d candidates, accepted %d, rejected %d",
                len(all_candidates),
                len(accepted_links),
                len(rejected),
            )
        except Exception as e:
            logger.warning("[PLP→PDP][CR-E2E-003A] Failed to save link collection evidence: %s", e, exc_info=True)
            summary["error"] = str(e)

        return summary
