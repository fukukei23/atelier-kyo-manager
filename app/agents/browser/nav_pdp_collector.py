# ==============================================================================
# File: app/agents/browser/nav_pdp_collector.py
# Purpose: NavPdpCollectorMixin - PDP link collection and evidence saving
# ==============================================================================
"""
NavigationDriver から PDP リンク収集ロジックを抽出した Mixin。

抽出対象メソッド:
- collect_pdp_links
- _save_link_collection_evidence
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

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

logger = logging.getLogger(__name__)


def _dedupe_keep_order(items: list[str]) -> list[str]:
    """重複を削除しつつ順序を保持する"""
    return list(dict.fromkeys([i for i in (items or []) if i]))


class NavPdpCollectorMixin:
    """PDP リンク収集と証跡保存を担当する Mixin。"""

    async def collect_pdp_links(
        self,
        ctx: NavigationContext,
    ) -> list[str]:
        """
        旧 BrowserUseAgent._collect_pdp_links のロジック。

        Phase 1a: Global <a href> sweep + Regex Filter
        Phase 1b: Selector-based補完
        Phase 2: Deep Extraction Fallback (only if Phase 1 failed)
        Phase 3: Noise Filtering & Saving

        Args:
            ctx: ナビゲーションコンテキスト

        Returns:
            List[str]: PDP リンクのリスト
        """
        page: Page = self.page  # type: ignore[attr-defined]
        site_config = ctx.site_config
        run_context = ctx.run_context
        target_url = page.url
        found_links: set[str] = set()

        all_candidates: list[LinkCandidate] = []

        # Phase 1a: Global <a href> sweep + Regex Filter
        try:
            raw_hrefs: list[str] = await page.evaluate(
                "() => Array.from(document.querySelectorAll('a[href]')).map(a => a.getAttribute('href')).filter(Boolean)"
            )
        except Exception as e:
            logger.warning(f"[PLP→PDP][1a] Sweep failed: {e}")
            raw_hrefs = []
        pdp_rx = re.compile(r"/(products?|p)/", re.I)
        html_pdp_rx = re.compile(r"/(?:en-int|en-jp/en-int)/[^/]+/[^/]+/[^/]+\.html", re.I)
        phase1a_candidates: list[LinkCandidate] = []
        for href in raw_hrefs:
            if "onetrust.com" in href.lower():
                continue
            if pdp_rx.search(href) or html_pdp_rx.search(href):
                norm_url, norm_info = normalize_candidate_url(href, target_url, site_config)
                candidate = LinkCandidate(
                    url=href,
                    phase="1a",
                    normalized_url=norm_url,
                    source_selector="global_sweep",
                )
                candidate = classify_candidate(candidate, target_url, site_config)
                if candidate.product_url_rules:
                    candidate.product_url_rules["normalization_info"] = norm_info
                phase1a_candidates.append(candidate)
                all_candidates.append(candidate)
                if candidate.accepted:
                    found_links.add(norm_url)
        if found_links:
            logger.info(f"[PLP→PDP][1a] Sweep found {len(found_links)} links.")

        # Phase 1b: Selector-based補完
        plp_selectors = (site_config.get("selectors", {}) or {}).get("plp", {}) or {}
        pdp_selectors = (site_config.get("selectors", {}) or {}).get("pdp", {}) or {}

        pdp_link_selectors = _dedupe_keep_order(
            (plp_selectors.get("pdp_link_selectors", []) or []) + (pdp_selectors.get("pdp_link_selectors", []) or [])
        )

        if not pdp_link_selectors:
            pdp_link_selectors = [
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
            ]

        PLP_PDP_LINK_SELECTORS = pdp_link_selectors
        phase1b_candidates: list[LinkCandidate] = []
        for sel in PLP_PDP_LINK_SELECTORS:
            try:
                nodes = await page.query_selector_all(sel)
                if not nodes:
                    continue
                matched_count = 0
                rejected_count = 0
                for n in nodes:
                    href = (
                        await n.get_attribute("href")
                        or await n.get_attribute("data-href")
                        or await n.get_attribute("data-product-url")
                        or await n.get_attribute("data-url")
                    )
                    if not href:
                        candidate = LinkCandidate(
                            url="",
                            phase="1b",
                            normalized_url="",
                            reject_reasons=[RejectReason.NO_HREF.value],
                            source_selector=sel,
                        )
                        phase1b_candidates.append(candidate)
                        all_candidates.append(candidate)
                        rejected_count += 1
                        continue
                    if "onetrust.com" in href.lower():
                        continue
                    norm_url, norm_info = normalize_candidate_url(href, target_url, site_config)
                    candidate = LinkCandidate(
                        url=href,
                        phase="1b",
                        normalized_url=norm_url,
                        source_selector=sel,
                    )
                    candidate = classify_candidate(candidate, target_url, site_config)
                    if candidate.product_url_rules:
                        candidate.product_url_rules["normalization_info"] = norm_info
                    phase1b_candidates.append(candidate)
                    all_candidates.append(candidate)
                    if candidate.accepted:
                        found_links.add(norm_url)
                        matched_count += 1
                    else:
                        rejected_count += 1
                if matched_count > 0:
                    logger.info(f"[PLP→PDP][1b] selector='{sel}' added {matched_count} links.")
                elif nodes and rejected_count > 0:
                    logger.warning(
                        f"[PLP→PDP][1b] selector='{sel}' found {len(nodes)} elements, "
                        f"but {rejected_count} were rejected."
                    )
            except Exception as e:
                logger.warning(f"[PLP→PDP][1b] selector='{sel}' failed: {e}")

        # Phase 2: Deep Extraction Fallback (only if Phase 1 failed)
        phase2_candidates: list[LinkCandidate] = []
        if not found_links:
            logger.warning("[PLP→PDP] Phase 1a/1b found no links. Falling back to Phase 2 (Deep Extraction)...")
            try:
                deep_hrefs = await self._run_deep_extraction_phase2(page, site_config)  # type: ignore[attr-defined]
                for href in deep_hrefs:
                    norm_url, norm_info = normalize_candidate_url(href, target_url, site_config)
                    candidate = LinkCandidate(
                        url=href,
                        phase="2",
                        normalized_url=norm_url,
                        source_selector="deep_extraction",
                    )
                    candidate = classify_candidate(candidate, target_url, site_config)
                    if candidate.product_url_rules:
                        candidate.product_url_rules["normalization_info"] = norm_info
                    phase2_candidates.append(candidate)
                    all_candidates.append(candidate)
                    if candidate.accepted:
                        found_links.add(norm_url)
                if found_links:
                    logger.info(f"[PLP→PDP][2] Deep Extraction found {len(found_links)} links.")
            except Exception as e:
                logger.error(f"[PLP→PDP][2] Deep Extraction failed: {e}")

        # Evidence-based relaxed filtering
        if not found_links and all_candidates:
            refilter_config = (site_config or {}).get("pdp_link_refilter", {})
            if refilter_config.get("enabled", False):
                logger.info(
                    f"[PLP→PDP][Refilter] {len(all_candidates)} candidates found but all rejected. "
                    "Attempting evidence-based relaxed filtering..."
                )

                for candidate in all_candidates:
                    if not candidate.normalized_url:
                        continue
                    if (
                        candidate.product_url_rules
                        and not candidate.product_url_rules.get("forbidden_path_matched", True)
                        and candidate.product_url_rules.get("domain_allowed", False)
                        and candidate.source_selector
                        and any(
                            keyword in candidate.source_selector.lower()
                            for keyword in ["product", "card", "tile", "item"]
                        )
                    ):
                        found_links.add(candidate.normalized_url)
                        candidate.accepted = True
                        candidate.reject_reasons = [
                            r for r in candidate.reject_reasons if r not in (RejectReason.NO_PRODUCTS_PATH.value,)
                        ]
                        candidate.notes = (candidate.notes or "") + " [Refilter: product card source]"
                        logger.debug(
                            f"[PLP→PDP][Refilter] Accepted candidate (product card source): {candidate.normalized_url}"
                        )

                if not found_links and refilter_config.get("allow_same_site_only", False):
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
                                candidate.notes = (candidate.notes or "") + " [Refilter: same-site]"
                                logger.debug(
                                    f"[PLP→PDP][Refilter] Accepted same-site candidate: {candidate.normalized_url}"
                                )

                if not found_links:
                    ignore_reasons = refilter_config.get("ignore_reject_reasons", [])
                    if ignore_reasons:
                        for candidate in all_candidates:
                            if not candidate.normalized_url:
                                continue
                            if candidate.product_url_rules and not candidate.product_url_rules.get(
                                "forbidden_path_matched", False
                            ):
                                filtered_reasons = [r for r in candidate.reject_reasons if r not in ignore_reasons]
                                if not filtered_reasons:
                                    found_links.add(candidate.normalized_url)
                                    candidate.accepted = True
                                    candidate.reject_reasons = []
                                    candidate.notes = (candidate.notes or "") + f" [Refilter: ignored {ignore_reasons}]"
                                    logger.debug(
                                        f"[PLP→PDP][Refilter] Accepted candidate (ignored {ignore_reasons}): {candidate.normalized_url}"
                                    )

                if found_links:
                    logger.info(
                        f"[PLP→PDP][Refilter] Evidence-based relaxed filtering accepted {len(found_links)} links."
                    )
                else:
                    logger.warning(
                        "[PLP→PDP][Refilter] No candidates accepted after relaxed filtering. "
                        "Consider click fallback or selector adjustment."
                    )

        links = sorted(list(found_links))
        if not links:
            log_file_path = None
            if ctx and ctx.run_context:
                log_file_path = ctx.run_context.get_path("system.log")
            logger.error(
                f"[PLP→PDP] No PDP hrefs found after all phases. "
                f"Found {len(found_links)} links total, {len(all_candidates)} candidates collected. "
                f"Target URL: {target_url}. "
                "This indicates a selector mismatch or URL validation failure."
            )
            if log_file_path and log_file_path.exists():
                logger.error(f"[PLP→PDP] Full error log available at: {log_file_path}")
            elif ctx and ctx.run_context:
                logger.error(f"[PLP→PDP] Error log will be saved to: {ctx.run_context.get_path('system.log')}")

            try:
                total_elements_found = 0
                for sel in PLP_PDP_LINK_SELECTORS:
                    try:
                        nodes = await page.query_selector_all(sel)
                        if nodes:
                            total_elements_found += len(nodes)
                            if total_elements_found == len(nodes):
                                sample_node = nodes[0]
                                sample_attrs = {
                                    "tag": await sample_node.evaluate("el => el.tagName"),
                                    "href": await sample_node.get_attribute("href"),
                                    "data-href": await sample_node.get_attribute("data-href"),
                                    "data-product-url": await sample_node.get_attribute("data-product-url"),
                                    "class": await sample_node.get_attribute("class"),
                                }
                                logger.debug(f"[PLP→PDP] Sample element from selector '{sel}': {sample_attrs}")
                    except Exception:
                        pass
                if total_elements_found > 0:
                    logger.warning(
                        f"[PLP→PDP] Found {total_elements_found} elements matching selectors, "
                        "but none passed URL validation or product URL check."
                    )
            except Exception as e:
                logger.debug(f"[PLP→PDP] Diagnostic info collection failed: {e}")

        # Phase 3: Noise Filtering & Saving
        cleaned: list[str] = []
        noise_rx = re.compile(r"/(collections?|seasons?|client-service|login|legal|cart|wishlist|search)/", re.I)
        for u in links:
            if not noise_rx.search(u):
                cleaned.append(u)
        logger.info(f"[PLP→PDP] collected {len(cleaned)} PDP-like links (raw={len(links)})")

        link_collection_summary = await self._save_link_collection_evidence(
            all_candidates=all_candidates,
            accepted_links=cleaned,
            run_context=run_context,
            ctx=ctx,
        )

        ctx.link_collection_summary = link_collection_summary

        try:
            sample = cleaned[:20]
            logger.debug(f"[PLP→PDP] sample={sample}")
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
            logger.debug(f"[PLP→PDP] TelemetryClient.save_raw_hrefs failed: {e}")
        return cleaned

    async def _save_link_collection_evidence(
        self,
        all_candidates: list[LinkCandidate],
        accepted_links: list[str],
        run_context: Any | None,
        ctx: NavigationContext,
    ) -> dict[str, Any]:
        """
        CR-E2E-003A: リンク収集の証跡を保存し、サマリを返す。

        Args:
            all_candidates: すべての候補リスト
            accepted_links: 受け入れられたリンクリスト
            run_context: RunContext（任意）
            ctx: NavigationContext

        Returns:
            Dict[str, Any]: サマリデータ
        """
        summary: dict[str, Any] = {
            "total_candidates": len(all_candidates),
            "total_valid": len(accepted_links),
            "top_reject_reasons": {},
            "sample_candidates": [],
        }

        if not run_context or not hasattr(run_context, "save_json"):
            return summary

        validation_report: dict[str, Any] | None = None

        try:
            phase1_candidates: list[dict[str, Any]] = []
            phase2_candidates: list[dict[str, Any]] = []
            phase3_candidates: list[dict[str, Any]] = []

            for candidate in all_candidates[:200]:
                candidate_dict = {
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

                if candidate.phase in ("1a", "1b"):
                    phase1_candidates.append(candidate_dict)
                elif candidate.phase == "2":
                    phase2_candidates.append(candidate_dict)
                elif candidate.phase == "3":
                    phase3_candidates.append(candidate_dict)

            run_context.save_json(
                "pdp_link_candidates_phase1.json",
                {"phase": "1a/1b", "candidates": phase1_candidates, "total": len(phase1_candidates)},
            )

            run_context.save_json(
                "pdp_link_candidates_phase2.json",
                {"phase": "2", "candidates": phase2_candidates, "total": len(phase2_candidates)},
            )

            if phase3_candidates:
                run_context.save_json(
                    "pdp_link_candidates_phase3.json",
                    {"phase": "3", "candidates": phase3_candidates, "total": len(phase3_candidates)},
                )

            reject_reason_counts: dict[str, int] = {}
            for candidate in all_candidates:
                for reason in candidate.reject_reasons:
                    reject_reason_counts[reason] = reject_reason_counts.get(reason, 0) + 1

            top_reject_reasons = dict(sorted(reject_reason_counts.items(), key=lambda x: x[1], reverse=True)[:10])

            domain_rejected_count = 0
            path_rejected_count = 0
            allow_path_matched_count = 0
            forbidden_path_matched_count = 0
            allow_path_pattern_counts: dict[str, int] = {}
            forbidden_path_pattern_counts: dict[str, int] = {}

            for candidate in all_candidates:
                if candidate.product_url_rules:
                    rules = candidate.product_url_rules
                    if not rules.get("domain_allowed", True):
                        domain_rejected_count += 1
                    if rules.get("forbidden_path_matched", False):
                        forbidden_path_matched_count += 1
                        pattern = rules.get("forbidden_path_pattern")
                        if pattern:
                            forbidden_path_pattern_counts[pattern] = forbidden_path_pattern_counts.get(pattern, 0) + 1
                    if rules.get("allow_path_matched", False):
                        allow_path_matched_count += 1
                        pattern = rules.get("allow_path_pattern")
                        if pattern:
                            allow_path_pattern_counts[pattern] = allow_path_pattern_counts.get(pattern, 0) + 1
                    if not rules.get("allow_path_matched", False) and not rules.get("forbidden_path_matched", False):
                        path_rejected_count += 1

            validation_report = {
                "total_candidates": len(all_candidates),
                "total_valid": len(accepted_links),
                "total_rejected": len(all_candidates) - len(accepted_links),
                "reject_reason_counts": reject_reason_counts,
                "top_reject_reasons": top_reject_reasons,
                "domain_rejected_count": domain_rejected_count,
                "path_rejected_count": path_rejected_count,
                "allow_path_matched_count": allow_path_matched_count,
                "forbidden_path_matched_count": forbidden_path_matched_count,
                "allow_path_pattern_counts": dict(
                    sorted(allow_path_pattern_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                ),
                "forbidden_path_pattern_counts": dict(
                    sorted(forbidden_path_pattern_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                ),
                "all_rejected_candidates": [
                    {
                        "raw_href": c.url,
                        "resolved_url": c.normalized_url,
                        "origin": c.origin or "",
                        "reject_reasons": c.reject_reasons,
                        "phase": c.phase,
                        "source_selector": c.source_selector or "",
                        "notes": c.notes or "",
                        "product_url_rules": c.product_url_rules if c.product_url_rules else {},
                    }
                    for c in all_candidates
                    if not c.accepted
                ][:200],
                "sample_rejected": [
                    {
                        "raw_href": c.url,
                        "resolved_url": c.normalized_url,
                        "origin": c.origin or "",
                        "reject_reasons": c.reject_reasons,
                        "phase": c.phase,
                        "source_selector": c.source_selector or "",
                        "notes": c.notes or "",
                        "product_url_rules": c.product_url_rules if c.product_url_rules else {},
                    }
                    for c in all_candidates
                    if not c.accepted
                ][:10],
            }

            try:
                run_context.save_json("pdp_link_validation_report.json", validation_report)
            except Exception as e:
                logger.error(f"[PLP→PDP] Failed to save validation report: {e}", exc_info=True)
                try:
                    report_path = run_context.get_path("pdp_link_validation_report.json")
                    with open(report_path, "w", encoding="utf-8") as f:
                        json.dump(validation_report, f, indent=2, ensure_ascii=False)
                except Exception as e2:
                    logger.error(f"[PLP→PDP] Failed to write validation report directly: {e2}", exc_info=True)

            summary["total_candidates"] = len(all_candidates)
            summary["total_valid"] = len(accepted_links)
            summary["top_reject_reasons"] = top_reject_reasons
            summary["sample_candidates"] = [
                {
                    "raw_href": c.url,
                    "resolved_url": c.normalized_url,
                    "origin": c.origin or "",
                    "reject_reasons": c.reject_reasons,
                    "phase": c.phase,
                    "source_selector": c.source_selector or "",
                    "passed": c.accepted,
                }
                for c in all_candidates[:10]
            ]

            logger.info(
                f"[PLP→PDP][CR-E2E-003A] Collected {len(all_candidates)} candidates, "
                f"accepted {len(accepted_links)}, rejected {len(all_candidates) - len(accepted_links)}"
            )
        except Exception as e:
            logger.warning(f"[PLP→PDP][CR-E2E-003A] Failed to save link collection evidence: {e}", exc_info=True)
            summary["error"] = str(e)
        finally:
            if run_context and validation_report is not None:
                try:
                    report_path = run_context.get_path("pdp_link_validation_report.json")
                    with open(report_path, "w", encoding="utf-8") as f:
                        json.dump(validation_report, f, indent=2, ensure_ascii=False)
                    logger.info(f"[PLP→PDP] Saved validation report (finally) to: {report_path}")
                except Exception as e:
                    logger.error(f"[PLP→PDP] Failed to save validation report in finally: {e}", exc_info=True)

        return summary
