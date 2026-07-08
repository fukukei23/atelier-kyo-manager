#!/usr/bin/env python3
"""Site runner script for BrowserUseAgent with site alias support."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents.browser_use_agent import BrowserUseAgent
from app.core.run_context import RunContext
from app.models.result_models import DiscoveryResult
from app.utils.observability import write_fail_snapshot

try:
    from app.config.loader import get_site_config, load_full_config
    from app.config.protocols import DefaultSiteConfigProvider
except ImportError:
    load_full_config = None
    get_site_config = None
    DefaultSiteConfigProvider = None

logger: logging.Logger = logging.getLogger(__name__)


# ── Types ────────────────────────────────────────────────


class ProxyMode(str, Enum):
    AUTO = "auto"
    ON = "on"
    OFF = "off"


# ── Constants ────────────────────────────────────────────

SITE_ALIASES: dict[str, str] = {
    "moncler": "MONCLER_OFFICIAL",
    "moncler_official": "MONCLER_OFFICIAL",
    "moncler-official": "MONCLER_OFFICIAL",
}


def resolve_site_name(site_input: str) -> str:
    site_upper = site_input.upper()
    if site_upper in SITE_ALIASES.values():
        return site_upper
    site_lower = site_input.lower()
    if site_lower in SITE_ALIASES:
        return SITE_ALIASES[site_lower]

    for loader in (
        lambda: DefaultSiteConfigProvider() if DefaultSiteConfigProvider else None,
        lambda: load_full_config() if load_full_config else None,
    ):
        try:
            provider = loader()
            if not provider:
                continue
            full_config = provider.get_full_config() if hasattr(provider, "get_full_config") else provider
            for name in full_config or {}:
                if name.upper() == site_upper:
                    return name.upper()
        except Exception:
            pass

    return site_upper


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run BrowserUseAgent for a specific site with alias support.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m app.scripts.run_site moncler --dry-run
  python -m app.scripts.run_site moncler --query "down jacket" --headful
  python -m app.scripts.run_site MONCLER_OFFICIAL --url "https://www.moncler.com/..." --dry-run
        """,
    )
    p.add_argument("site", help="Site name or alias")
    p.add_argument("--query", default="down jacket", help="Query/brand label")
    p.add_argument("--url", help="Target URL (PLP/PDP)")
    p.add_argument("--dry-run", action="store_true", help="Validate config only")

    head = p.add_mutually_exclusive_group()
    head.add_argument("--headless", dest="headful", action="store_false", help="Headless (default)")
    head.add_argument("--headful", dest="headful", action="store_true", help="Headful browser")
    p.set_defaults(headful=False)

    p.add_argument("--enable-video", action="store_true", help="Enable video recording")
    p.add_argument("--timeout", type=int, default=60, help="Timeout seconds (default: 60)")

    proxy = p.add_mutually_exclusive_group()
    proxy.add_argument("--proxy-mode", choices=[m.value for m in ProxyMode], default=ProxyMode.AUTO.value)
    proxy.add_argument(
        "--use-proxy", action="store_true", dest="proxy_mode_on", help="[Deprecated] Use --proxy-mode on"
    )

    p.add_argument("--human-like", action="store_true", help="Human-like cursor/scroll")
    return p.parse_args()


# ── Setup helpers ───────────────────────────────────────


def _setup_instance_dirs(site_name: str) -> Path:
    instance_dir = ROOT / "instance" / site_name.lower().replace("_", "-")
    instance_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("cache", "cookies", "logs"):
        (instance_dir / sub).mkdir(exist_ok=True)

    last_run_json = instance_dir / "last_run.json"
    if not last_run_json.exists():
        from datetime import datetime, timezone

        last_run_json.write_text(
            json.dumps(
                {
                    "site": site_name,
                    "strategy_version": f"{site_name.lower().replace('_', '-')}-latest",
                    "last_run_at": None,
                    "last_run_id": None,
                    "last_status": None,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return instance_dir


def _load_site_config(site_name: str) -> tuple[Any | None, dict[str, Any] | None]:
    """Returns (config_provider, site_config). Either may be None on failure."""
    if DefaultSiteConfigProvider is not None:
        try:
            provider = DefaultSiteConfigProvider()
            return provider, provider.get_site_config(site_name)
        except Exception as e:
            logger.error("[run_site] SiteConfigProvider failed: %s", e, exc_info=True)
            return None, None

    if get_site_config:
        try:
            return None, get_site_config(site_name)
        except Exception as e:
            logger.error("[run_site] Config loader failed: %s", e, exc_info=True)
            return None, None

    logger.error("[run_site] No config loader available.")
    return None, None


def _resolve_target_url(args: argparse.Namespace, site_config: dict[str, Any], site_name: str) -> str | None:
    url = args.url or site_config.get("home_url") or site_config.get("discovery_settings", {}).get("home_url")
    if not url:
        logger.error("[run_site] No target URL for '%s'", site_name)
    return url


def _resolve_proxy(args: argparse.Namespace) -> bool | None:
    mode = ProxyMode(args.proxy_mode)
    if mode != ProxyMode.AUTO:
        return mode == ProxyMode.ON
    if getattr(args, "proxy_mode_on", False):
        return True
    return None


# ── Dry-run validation ─────────────────────────────────


def _check_import_available(label: str, module_path: str, attr: str, required: bool = False) -> bool:
    try:
        mod = __import__(module_path, fromlist=[attr])
        getattr(mod, attr)
        logger.info("[run_site]   %s: available", label)
        return True
    except (ImportError, AttributeError) as e:
        (logger.error if required else logger.warning)(
            "[run_site]   %s: %s - %s",
            label,
            "FAILED" if required else "not available",
            e,
        )
        return not required


def _validate_dry_run(site_name: str, site_config: dict[str, Any], target_url: str, args: argparse.Namespace) -> bool:
    logger.info("[run_site] DRY-RUN validation...")

    _check_import_available("Stealth", "scraping.stealth", "apply_stealth_to_context")
    _check_import_available("SessionManager", "app.agents.browser.session_manager", "SessionManager", required=True)
    _check_import_available("BrowserUseAgent", "app.agents.browser_use_agent", "BrowserUseAgent", required=True)

    if site_name == "MONCLER_OFFICIAL":
        _check_import_available("Moncler patch", "app.agents.browser_use_moncler_patch", "moncler_plp_recovery")

    missing = [k for k in ("discovery_settings", "selectors") if k not in site_config]
    if missing:
        logger.warning("[run_site]   Config missing keys: %s", missing)
    else:
        logger.info("[run_site]   Config structure: valid")

    logger.info("[run_site]   DRY-RUN complete. site=%s, query=%s, url=%s", site_name, args.query, target_url)
    return True


# ── Result handling ────────────────────────────────────


def _result_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_dict"):
        return result.to_dict()
    if hasattr(result, "__dict__"):
        return {k: v for k, v in result.__dict__.items() if not k.startswith("_")}
    return {
        "ok": getattr(result, "ok", False),
        "site": getattr(result, "site", ""),
        "query": getattr(result, "query", ""),
    }


def _save_result(run_ctx: RunContext, result: Any) -> None:
    result_dict = _result_to_dict(result)

    if isinstance(result_dict, dict) and "evidence" in result_dict:
        ev = result_dict["evidence"]
        if "success_stage" in ev:
            result_dict["success_stage"] = ev["success_stage"]
        if "criteria" in ev:
            result_dict["criteria"] = ev["criteria"]

    result_path = run_ctx.get_path("result.json")
    result_path.write_text(json.dumps(result_dict, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = getattr(result, "ok", False)
    log_path = run_ctx.get_path("system.log")
    logger.info("[run_site] ok=%s run_dir=%s result=%s", ok, run_ctx.run_path, result_path)
    if not ok:
        logger.error("[run_site] Failed. Log: %s Snapshot: %s/failure_dom.html", log_path, run_ctx.run_path)


async def _handle_timeout(
    run_ctx: RunContext,
    agent: BrowserUseAgent,
    timeout_sec: int,
    site_config: dict[str, Any],
    pending: set[asyncio.Task[Any]],
) -> DiscoveryResult:
    logger.error("[run_site] Timeout after %ds", timeout_sec)
    try:
        await write_fail_snapshot(
            run_ctx,
            getattr(agent, "_page", None),
            getattr(getattr(agent, "_page", None), "url", None),
            TimeoutError(f"Timeout after {timeout_sec}s"),
            site_config,
        )
    except Exception as e:
        logger.error("[run_site] write_fail_snapshot failed: %s", e)

    for t in pending:
        t.cancel()
        with contextlib.suppress(Exception):
            await t

    return DiscoveryResult(ok=False, message=f"Timeout after {timeout_sec}s", evidence={"timeout_sec": timeout_sec})


# ── Main ───────────────────────────────────────────────


async def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    site_name = resolve_site_name(args.site)
    logger.info("[run_site] '%s' -> '%s'", args.site, site_name)

    _setup_instance_dirs(site_name)
    config_provider, site_config = _load_site_config(site_name)
    if not site_config:
        try:
            src = config_provider or (DefaultSiteConfigProvider() if DefaultSiteConfigProvider else load_full_config)
            avail = list(src.get_full_config().keys() if hasattr(src, "get_full_config") else src.keys())
        except Exception:
            avail = []
        logger.error("[run_site] Config not found for '%s'. Available: %s", site_name, avail)
        return 1

    target_url = _resolve_target_url(args, site_config, site_name)
    if not target_url:
        return 1

    if args.dry_run:
        return 0 if _validate_dry_run(site_name, site_config, target_url, args) else 1

    # Apply CLI overrides
    ds = site_config.setdefault("discovery_settings", {})
    ds["timeout_sec"] = args.timeout
    if args.enable_video:
        ds["enable_video"] = True
    if args.human_like:
        ds["enable_human_like"] = True

    run_ctx = RunContext()
    try:
        handler = run_ctx.setup_system_logger()
        logging.getLogger().addHandler(handler)
        logger.info("[run_site] System log: %s", run_ctx.get_path("system.log"))
    except Exception as e:
        logger.warning("[run_site] System logger failed: %s", e, exc_info=True)

    proxy = _resolve_proxy(args)
    runtime_kwargs: dict[str, Any] = {
        "headless": not args.headful,
        "timeout_sec": args.timeout,
        "enable_video": args.enable_video,
        "site": site_name,
        "enable_human_like": args.human_like,
    }
    if proxy is not None:
        runtime_kwargs["use_proxy"] = proxy

    agent = BrowserUseAgent(runtime_kwargs=runtime_kwargs, config_provider=config_provider)
    timeout_sec = args.timeout + 60
    run_task = asyncio.create_task(
        agent.run(
            site=site_name,
            query=args.query,
            site_config=site_config,
            run_context=run_ctx,
            target_url=target_url,
            likely_plp=True,
        )
    )

    done, pending = await asyncio.wait({run_task}, timeout=timeout_sec)
    if not done:
        result = await _handle_timeout(run_ctx, agent, timeout_sec, site_config, pending)
        result.site = site_name
        result.query = args.query
    else:
        result = run_task.result()

    _save_result(run_ctx, result)
    return 0 if getattr(result, "ok", False) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
