# -*- coding: utf-8 -*-
"""
settings.py - Settings resolution and time budget management for BrowserUseAgent

This module handles:
- Run settings resolution from site_config and runtime_kwargs
- Time budget management (watchdog, time left calculation)
- Configuration constants (VIEWPORT_POOL, USER_AGENT_POOL, SESSION_DIR, etc.)

Note: Browser session management logic (route setup, init scripts, session restore)
      has been moved to SessionManager (app/agents/browser/session_manager.py).
"""

from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

from app.agents.browser.extractor import PDPSizeSelectPolicy, DEFAULT_PDP_PARALLEL_LIMIT
from app.core.run_context import RunContext

OVERALL_PLP_BUDGET_MS_DEFAULT = 120000  # 120s watchdog

# Viewport and User-Agent pools for rotation
VIEWPORT_POOL = [
    {"width": 1920, "height": 1080},
    {"width": 1366, "height": 768},
    {"width": 1536, "height": 864},
]

USER_AGENT_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
]

SESSION_DIR = Path("instance/sessions")


def resolve_run_settings(
    site_config: Dict[str, Any],
    runtime_kwargs: Dict[str, Any],
    logger: Any,
) -> Dict[str, Any]:
    """
    Resolve run settings from site_config and runtime_kwargs.
    
    Priority: runtime_kwargs > site_config > defaults
    """
    ds = site_config.get("discovery_settings", {}) or {}
    site_key_guess = (
        site_config.get("site_key")
        or site_config.get("id")
        or runtime_kwargs.get("site")
        or ""
    )
    site_key_guess = str(site_key_guess or "").upper()
    vrt = ds.get("vrt", {}) or {}
    
    logger.info(f"[Debug] runtime enable_video flag: {runtime_kwargs.get('enable_video')}")
    enable_har = runtime_kwargs.get("enable_har", ds.get("enable_har", True))
    enable_trace = runtime_kwargs.get("enable_trace", ds.get("enable_trace", True))

    # --- enable_video resolution (CLI > site config > env > default) ---
    cli_enable_video = runtime_kwargs.get("enable_video")
    cfg_enable_video = ds.get("enable_video")
    env_raw = os.getenv("ATK_ENABLE_VIDEO") or os.getenv("ENABLE_VIDEO")
    env_enable_video: Optional[bool] = None
    if env_raw is not None:
        env_enable_video = str(env_raw).strip().lower() in ("1", "true", "yes", "y", "on")

    if cli_enable_video is not None:
        enable_video = bool(cli_enable_video)
    elif cfg_enable_video is not None:
        enable_video = bool(cfg_enable_video)
    elif env_enable_video is not None:
        enable_video = env_enable_video
    else:
        enable_video = True if site_key_guess == "MONCLER_OFFICIAL" else False

    default_accept_language = "en-GB,en;q=0.8"
    if site_key_guess == "MONCLER_OFFICIAL":
        default_accept_language = "en-US,en;q=0.8"

    def _dedupe_keep_order(items: list) -> list:
        return list(dict.fromkeys([i for i in (items or []) if i]))

    settings = {
        "timeout_sec": runtime_kwargs.get("timeout_sec") or ds.get("timeout_sec", 60),
        "headless": runtime_kwargs.get("headless", True),
        "slow_mo": runtime_kwargs.get("slow_mo", 0),
        "viewport": ds.get("viewport"),
        "user_agent": ds.get("user_agent"),
        "extra_http_headers": ds.get("extra_http_headers"),
        "accept_language": ds.get("accept_language", default_accept_language),
        "enable_har": enable_har,
        "enable_trace": enable_trace,
        "enable_video": enable_video,
        "enable_locale_escape": bool(ds.get("enable_locale_escape", True)),
        "overall_plp_budget_ms": int(ds.get("overall_plp_budget_ms", OVERALL_PLP_BUDGET_MS_DEFAULT)),
        "pdp_parallel_limit": int(ds.get("pdp_parallel_limit", DEFAULT_PDP_PARALLEL_LIMIT)),
        "pdp_retry_once": bool(ds.get("pdp_retry_once", True)),
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
        "wait_for_selectors": _dedupe_keep_order(ds.get("wait_for_selectors") or []),
        "wait_until": ds.get("wait_until") or "domcontentloaded",
        "plp_scroll_rounds": int(ds.get("plp_scroll_rounds", 10)),
        "extra_block_routes": _dedupe_keep_order(ds.get("extra_block_routes") or []),
        "pdp_price_wait_ms": int(ds.get("pdp_price_wait_ms", 4000)),
        "locale_recover_max": int(ds.get("locale_recover_max", 5)),
        "enable_human_like": bool(runtime_kwargs.get("enable_human_like", ds.get("enable_human_like", False))),
        "enable_ua_rotation": bool(runtime_kwargs.get("enable_ua_rotation", ds.get("enable_ua_rotation", False))),
        "enable_viewport_rotation": bool(runtime_kwargs.get("enable_viewport_rotation", ds.get("enable_viewport_rotation", False))),
    }
    
    try:
        pdp_policy_cfg = ds.get("pdp_size_select_policy", {})
        settings["pdp_size_select_policy"] = PDPSizeSelectPolicy(
            mode=pdp_policy_cfg.get("mode", "off"),
            prefer_labels=pdp_policy_cfg.get("prefer_labels", [])
        )
    except Exception as e:
        logger.warning(f"Could not parse PDPSizeSelectPolicy: {e}. Defaulting to 'off'.")
        settings["pdp_size_select_policy"] = PDPSizeSelectPolicy()
    
    return settings


def start_watchdog(budget_ms: int) -> Tuple[float, int]:
    """Start time budget watchdog. Returns (start_time, budget_ms)."""
    return time.monotonic(), int(budget_ms)


def time_left_ms(start_t: float, budget_ms: int) -> int:
    """Calculate remaining time in milliseconds."""
    used = int((time.monotonic() - start_t) * 1000)
    return max(0, budget_ms - used)


def slice_timeout_ms(left_ms: int, cap_ms: int) -> int:
    """Slice timeout to fit within remaining budget and cap."""
    return max(500, min(left_ms, cap_ms))

