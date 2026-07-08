"""
Moncler PLP→PDP リンク抽出

CR-ATELIER-002: Moncler公式サイト専用のPLP→PDPリンク抽出・バリデーション。
extractor.py から分離。
"""

from __future__ import annotations

import contextlib
import logging
import re
from typing import Any
from urllib.parse import urljoin, urlparse

from playwright.async_api import Page

from app.agents.plugins.moncler_plp_v1 import (
    MONCLER_PLP_CONTAINER_SELECTORS,
    MONCLER_PLP_PDP_LINK_SELECTORS_PRIMARY,
)

logger = logging.getLogger(__name__)


async def extract_moncler_pdp_links(
    page: Page,
    ctx: Any,
    *,
    max_links: int = 50,
) -> list[str]:
    """
    CR-ATELIER-002 Step 3:
    MONCLER_OFFICIAL 専用の PLP→PDP 抽出ヘルパー。

    - DOM 構造は CR-ATELIER-002 Step3 の Spec / moncler_plp_v1.py のコメントに基づく。
    - PLP コンテナを特定し、コンテナ内の tile を query して、tile 内の <a href> から '/products/' を含む URL を収集
    - URL を正規化 / 重複排除 / バリデーション
    - CR-ATELIER-002 Step 3-3: raw_hrefs を収集し、reject_reason を集計、accepted==0 の場合 Telemetry に保存

    Args:
        page: Playwright Page オブジェクト
        ctx: NavigationContext または RunContext を含むコンテキスト
        max_links: 最大抽出リンク数

    Returns:
        List[str]: 有効な PDP URL のリスト
    """
    urls: list[str] = []
    target_url = page.url

    # site_config を取得（NavigationContext または dict から）
    site_config = {}
    run_context = None
    if hasattr(ctx, "site_config"):
        site_config = ctx.site_config or {}
        run_context = getattr(ctx, "run_context", None)
    elif isinstance(ctx, dict):
        site_config = ctx.get("site_config", {})
        run_context = ctx.get("run_context")

    logger.info(f"[PLP→PDP][Moncler] Starting extraction from URL: {target_url}")

    # CR-ATELIER-002 Step 3-3: raw_hrefs を収集
    raw_hrefs: list[str] = []
    rejection_stats: dict[str, int] = {
        "no_href": 0,
        "url_normalization_failed": 0,
        "external_domain": 0,
        "blocked_domain": 0,
        "double_locale_path": 0,
        "no_en_int_path": 0,
        "no_products_path": 0,
        "trap_pattern": 0,
        "other": 0,
    }

    # 1) PLP コンテナを特定（オプション、ログ用）
    for container_sel in MONCLER_PLP_CONTAINER_SELECTORS:
        try:
            container = await page.query_selector(container_sel)
            if container:
                logger.debug(f"[PLP→PDP][Moncler] Found container: {container_sel}")
                break
        except Exception:
            continue

    # CR-ATELIER-002 Step 5-2: セレクタ戦略のレイヤリング実装
    plp_selectors = (site_config.get("selectors", {}) or {}).get("plp", {}) or {}
    primary_selectors = plp_selectors.get("pdp_link_selectors", []) or MONCLER_PLP_PDP_LINK_SELECTORS_PRIMARY

    layer_stats: dict[str, int] = {
        "primary_raw": 0,
        "primary_accepted": 0,
        "secondary_raw": 0,
        "secondary_accepted": 0,
        "tertiary_raw": 0,
        "tertiary_accepted": 0,
    }

    # 2) Primary Layer で抽出を試みる
    raw_elements_count = 0
    current_selectors = primary_selectors

    for link_sel in current_selectors:
        try:
            nodes = await page.query_selector_all(link_sel)
            if not nodes:
                continue

            raw_elements_count += len(nodes)
            matched_count = 0
            rejected_count = 0

            for node in nodes:
                href = (
                    await node.get_attribute("href")
                    or await node.get_attribute("data-href")
                    or await node.get_attribute("data-product-url")
                    or await node.get_attribute("data-url")
                )

                if not href:
                    rejection_stats["no_href"] += 1
                    rejected_count += 1
                    continue

                raw_hrefs.append(href)

                try:
                    norm_url = urljoin(target_url, href)
                except Exception:
                    rejection_stats["url_normalization_failed"] += 1
                    rejected_count += 1
                    continue

                reject_reason = _get_moncler_rejection_reason(norm_url, target_url)
                if reject_reason:
                    rejection_stats[reject_reason] = rejection_stats.get(reject_reason, 0) + 1
                    rejected_count += 1
                    continue

                urls.append(norm_url)
                matched_count += 1

            if matched_count > 0:
                logger.info(
                    f"[PLP→PDP][Moncler] selector='{link_sel}' added {matched_count} links (rejected {rejected_count})"
                )
        except Exception as e:
            logger.debug(f"[PLP→PDP][Moncler] selector='{link_sel}' failed: {e}")

    # 3) 重複排除
    urls = list(dict.fromkeys(urls))

    # 4) max_links までに制限
    if len(urls) > max_links:
        urls = urls[:max_links]

    if raw_hrefs:
        sample_hrefs = raw_hrefs[:10]
        logger.debug(f"[PLP→PDP][Moncler] Raw hrefs (first 10): {sample_hrefs}")

    origin_rejected = rejection_stats.get("external_domain", 0) + rejection_stats.get("blocked_domain", 0)
    locale_rejected = rejection_stats.get("no_en_int_path", 0)
    path_rejected = rejection_stats.get("no_products_path", 0)
    trap_rejected = rejection_stats.get("trap_pattern", 0)
    other_rejected = (
        rejection_stats.get("no_href", 0)
        + rejection_stats.get("url_normalization_failed", 0)
        + rejection_stats.get("other", 0)
    )

    logger.info(
        f"[PLP→PDP][Moncler] Extraction summary: raw={len(raw_hrefs)}, "
        f"origin_rejected={origin_rejected}, "
        f"locale_rejected={locale_rejected}, "
        f"path_rejected={path_rejected}, "
        f"trap_rejected={trap_rejected}, "
        f"other_rejected={other_rejected}, "
        f"accepted={len(urls)}, "
        f"layer_stats={layer_stats}"
    )

    # accepted==0 の場合、Telemetry に保存
    if not urls and run_context:
        try:
            from app.agents.browser.telemetry import TelemetryContext

            telemetry = None
            if hasattr(ctx, "telemetry"):
                telemetry = ctx.telemetry
            elif hasattr(ctx, "run_context") and hasattr(ctx.run_context, "telemetry"):
                telemetry = ctx.run_context.telemetry

            if telemetry:
                tctx = TelemetryContext(
                    site=site_config.get("site_code") or site_config.get("site") or "MONCLER_OFFICIAL",
                    query=getattr(ctx, "query", None) or "",
                    run_id=getattr(run_context, "run_id", None),
                    stage="plp",
                )
                await telemetry.save_json(
                    "moncler_pdp_links_debug",
                    {
                        "raw_hrefs": raw_hrefs[:50],
                        "rejection_stats": {
                            "origin": origin_rejected,
                            "locale": locale_rejected,
                            "path": path_rejected,
                            "trap": trap_rejected,
                            "other": other_rejected,
                            "total_rejected": sum(rejection_stats.values()),
                        },
                        "rejection_details": rejection_stats,
                        "current_url": target_url,
                        "raw_elements_count": raw_elements_count,
                        "run_id": getattr(run_context, "run_id", None),
                    },
                    tctx,
                )
                logger.warning(
                    f"[PLP→PDP][Moncler] No valid PDP links found (raw={len(raw_hrefs)}, "
                    f"rejected={sum(rejection_stats.values())}), debug data saved to Telemetry"
                )
        except Exception as e:
            logger.debug(f"[PLP→PDP][Moncler] Failed to save Telemetry: {e}")

    logger.info(f"[PLP→PDP][Moncler] Collected {len(urls)} PDP links (from {raw_elements_count} raw elements)")

    layers_used: list[str] = []
    if layer_stats.get("primary_raw", 0) > 0 or layer_stats.get("primary_accepted", 0) > 0:
        layers_used.append("primary")
    if layer_stats.get("secondary_raw", 0) > 0 or layer_stats.get("secondary_accepted", 0) > 0:
        layers_used.append("secondary")
    if layer_stats.get("tertiary_raw", 0) > 0 or layer_stats.get("tertiary_accepted", 0) > 0:
        layers_used.append("tertiary")

    outcome_info = {
        "links": urls,
        "raw_count": len(raw_hrefs),
        "accepted_count": len(urls),
        "layer_stats": layer_stats,
        "layers_used": layers_used,
        "rejection_stats": rejection_stats,
        "current_url": target_url,
    }

    if isinstance(ctx, dict):
        ctx["moncler_outcome"] = outcome_info
    elif hasattr(ctx, "__dict__"):
        with contextlib.suppress(Exception):
            ctx.moncler_outcome = outcome_info

    return urls


def _get_moncler_rejection_reason(url: str, base_url: str) -> str | None:
    """CR-ATELIER-002 Step 3-3: Moncler URLバリデーションでrejectされた理由を取得"""
    try:
        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            return "other"

        host = parsed.netloc.lower()

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
                return "blocked_domain"

        if not host.endswith("moncler.com"):
            return "external_domain"

        path = parsed.path or ""

        double_locale_pattern = re.compile(r"/en-[a-z]{2}/en-int/", re.I)
        if double_locale_pattern.search(path):
            return "double_locale_path"

        if not path.startswith("/en-int/"):
            return "no_en_int_path"

        if "/products/" not in path and "/product/" not in path:
            return "no_products_path"

        trap_patterns = [
            "/404",
            "/not-found",
            "/search",
            "/legal/",
            "/client-service",
            "/collections/",
            "/seasons/",
            "/login",
            "/cart",
            "/wishlist",
        ]
        for trap_pattern in trap_patterns:
            if trap_pattern in path.lower():
                return "trap_pattern"

        return None
    except Exception:
        return "other"


def _is_valid_moncler_pdp_url(url: str, base_url: str) -> bool:
    """CR-ATELIER-002 Step 4-2: Moncler専用のURLバリデーション"""
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

        double_locale_pattern = re.compile(r"/en-[a-z]{2}/en-int/", re.I)
        if double_locale_pattern.search(path):
            return False

        if not path.startswith("/en-int/"):
            return False

        if "/products/" not in path and "/product/" not in path:
            return False

        trap_patterns = [
            "/404",
            "/not-found",
            "/search",
            "/legal/",
            "/client-service",
            "/collections/",
            "/seasons/",
            "/login",
            "/cart",
            "/wishlist",
        ]
        return all(trap_pattern not in path.lower() for trap_pattern in trap_patterns)
    except Exception:
        return False
