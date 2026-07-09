from __future__ import annotations

from typing import Any


def get_plp_config(site_config: dict[str, Any]) -> dict[str, Any]:
    """Return PLP configuration."""
    selectors = site_config.get("selectors") or {}
    plp_cfg = selectors.get("plp") or {}
    pdp_cfg = selectors.get("pdp") or {}
    discovery = site_config.get("discovery_settings", {}) or {}
    plp_discovery = discovery.get("plp") or {}
    return {
        "product_tiles": plp_cfg.get("product_tiles") or pdp_cfg.get("plp_container_selectors") or [],
        "product_link": plp_cfg.get("product_link") or pdp_cfg.get("pdp_link_selectors") or [],
        "container": plp_cfg.get("container") or pdp_cfg.get("plp_container_selectors") or ["main", "[role='main']"],
        "click_strategy": plp_cfg.get("click_strategy", "link"),
        "wait_for_navigation": plp_cfg.get("wait_for_navigation", True),
        "min_tiles": plp_cfg.get("min_tiles") or plp_discovery.get("min_tiles") or 8,
        "max_scroll_rounds": (
            plp_cfg.get("max_scroll_rounds")
            or plp_discovery.get("scroll_rounds")
            or discovery.get("plp_scroll_rounds")
            or 10
        ),
        "scroll_pause_ms": plp_cfg.get("scroll_pause_ms") or plp_discovery.get("scroll_pause_ms") or 160,
        "target_load_state": plp_cfg.get("target_load_state") or plp_discovery.get("wait_until") or "networkidle",
        "wait_for_selectors": plp_cfg.get("wait_for_selectors") or plp_discovery.get("wait_for_selectors") or [],
    }


def get_overlay_config(site_config: dict[str, Any]) -> dict[str, Any]:
    """Return overlay configuration."""
    nav_cfg = site_config.get("navigation") or {}
    overlays_cfg = nav_cfg.get("overlays") or {}
    selectors = site_config.get("selectors") or {}
    ui_cfg = selectors.get("ui") or {}
    return {
        "cookie": {
            "selectors": overlays_cfg.get("cookie_banner", {}).get("selectors") or ui_cfg.get("cookie_accept") or [],
            "wait_after_click_ms": overlays_cfg.get("cookie_banner", {}).get("wait_after_click_ms") or 500,
        },
        "geo": {
            "selectors": overlays_cfg.get("geo_popup", {}).get("selectors")
            or overlays_cfg.get("geo_modal_selectors")
            or [],
            "wait_after_click_ms": overlays_cfg.get("geo_popup", {}).get("wait_after_click_ms") or 500,
        },
        "other": overlays_cfg.get("other_overlays") or {},
    }


def get_trap_config(site_config: dict[str, Any]) -> dict[str, Any]:
    """Return trap detection configuration."""
    nav_cfg = site_config.get("navigation") or {}
    trap_cfg = nav_cfg.get("trap") or {}
    legacy_patterns = nav_cfg.get("trap_url_patterns") or []
    return {
        "detect_by_url": {
            "patterns": trap_cfg.get("detect_by_url", {}).get("patterns") or legacy_patterns or [],
            "exact_matches": trap_cfg.get("detect_by_url", {}).get("exact_matches") or [],
        },
        "detect_by_selector": trap_cfg.get("detect_by_selector") or [],
        "recovery_actions": trap_cfg.get("recovery_actions")
        or [
            {"action": "go_back", "max_attempts": 1},
            {"action": "goto_target", "target_url_key": "seed_plp_url", "max_attempts": 1},
        ],
    }
