#!/usr/bin/env python3
"""
Site runner script for BrowserUseAgent with site alias support.

Usage:
    python -m app.scripts.run_site moncler --dry-run
    python -m app.scripts.run_site moncler --query "down jacket" --headful
    python -m app.scripts.run_site MONCLER_OFFICIAL --url "https://www.moncler.com/..." --dry-run

Site aliases:
    - moncler -> MONCLER_OFFICIAL
    - moncler_official -> MONCLER_OFFICIAL
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import contextlib

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

logger = logging.getLogger(__name__)

# Site alias mapping
SITE_ALIASES: dict[str, str] = {
    "moncler": "MONCLER_OFFICIAL",
    "moncler_official": "MONCLER_OFFICIAL",
    "moncler-official": "MONCLER_OFFICIAL",
}


def resolve_site_name(site_input: str) -> str:
    """
    Resolve site alias to actual site name.

    Args:
        site_input: Site name or alias (e.g., "moncler", "MONCLER_OFFICIAL")

    Returns:
        Resolved site name (e.g., "MONCLER_OFFICIAL")
    """
    site_upper = site_input.upper()

    # Direct match
    if site_upper in SITE_ALIASES.values():
        return site_upper

    # Alias match
    site_lower = site_input.lower()
    if site_lower in SITE_ALIASES:
        return SITE_ALIASES[site_lower]

    # Try to find in config (SiteConfigProvider or loader)
    if DefaultSiteConfigProvider is not None:
        try:
            provider = DefaultSiteConfigProvider()
            full_config = provider.get_full_config()
            for site_name in full_config or {}:
                if site_name.upper() == site_upper:
                    return site_name.upper()
        except Exception:
            pass
    elif get_site_config and load_full_config:
        try:
            full_config = load_full_config()
            for site_name in full_config:
                if site_name.upper() == site_upper:
                    return site_name.upper()
        except Exception:
            pass

    # Return as-is (will be validated later)
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
    p.add_argument("site", help="Site name or alias (e.g., 'moncler', 'MONCLER_OFFICIAL')")
    p.add_argument("--query", default="down jacket", help="Query/brand label (default: 'down jacket')")
    p.add_argument("--url", help="Target URL (PLP/PDP). If not specified, uses site_config.home_url")
    p.add_argument("--dry-run", action="store_true", help="Dry-run mode (validate config and setup only)")

    head = p.add_mutually_exclusive_group()
    head.add_argument("--headless", dest="headful", action="store_false", help="Run with headless browser (default)")
    head.add_argument("--headful", dest="headful", action="store_true", help="Run with headful browser")
    p.set_defaults(headful=False)

    p.add_argument("--enable-video", action="store_true", help="Enable Playwright video recording")
    p.add_argument("--timeout", type=int, default=60, help="Timeout seconds (default: 60)")

    # Proxy mode: auto (default), on (force enable), off (force disable)
    proxy_group = p.add_mutually_exclusive_group()
    proxy_group.add_argument(
        "--proxy-mode",
        choices=["auto", "on", "off"],
        default="auto",
        help="Proxy mode: auto (use config/default), on (force enable), off (force disable)",
    )
    proxy_group.add_argument(
        "--use-proxy",
        action="store_true",
        dest="proxy_mode_on",
        help="[Deprecated] Use --proxy-mode on instead. Enable proxy usage if configured",
    )

    p.add_argument("--human-like", action="store_true", help="Enable human-like cursor/scroll pauses")

    return p.parse_args()


async def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Resolve site name from alias
    site_name = resolve_site_name(args.site)
    logger.info(f"[run_site] Resolved '{args.site}' -> '{site_name}'")

    # Ensure instance directory structure exists
    instance_dir = ROOT / "instance" / site_name.lower().replace("_", "-")
    instance_dir.mkdir(parents=True, exist_ok=True)
    (instance_dir / "cache").mkdir(exist_ok=True)
    (instance_dir / "cookies").mkdir(exist_ok=True)
    (instance_dir / "logs").mkdir(exist_ok=True)

    # Create last_run.json if it doesn't exist
    last_run_json = instance_dir / "last_run.json"
    if not last_run_json.exists():
        from datetime import datetime, timezone

        last_run_data = {
            "site": site_name,
            "strategy_version": f"{site_name.lower().replace('_', '-')}-latest",
            "last_run_at": None,
            "last_run_id": None,
            "last_status": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        last_run_json.write_text(json.dumps(last_run_data, indent=2), encoding="utf-8")
        logger.info(f"[run_site] Created {last_run_json}")

    # Load site config (SiteConfigProvider or loader)
    config_provider = None
    if DefaultSiteConfigProvider is not None:
        try:
            config_provider = DefaultSiteConfigProvider()
            site_config = config_provider.get_site_config(site_name)
        except Exception as e:
            logger.error(f"[run_site] Failed to load site_config via SiteConfigProvider: {e}", exc_info=True)
            return 1
    elif get_site_config:
        try:
            site_config = get_site_config(site_name)
        except Exception as e:
            logger.error(f"[run_site] Failed to load site_config: {e}", exc_info=True)
            return 1
    else:
        logger.error("[run_site] Failed to import config loader. Cannot load site configuration.")
        return 1

    if not site_config:
        try:
            avail = (
                (config_provider or DefaultSiteConfigProvider()).get_full_config().keys()
                if DefaultSiteConfigProvider
                else (load_full_config().keys() if load_full_config else [])
            )
        except Exception:
            avail = []
        logger.error(f"[run_site] Site config not found for '{site_name}'. Available sites: {list(avail)}")
        return 1
    logger.info(f"[run_site] Loaded site_config for '{site_name}'")

    # Determine target URL
    target_url = args.url
    if not target_url:
        target_url = site_config.get("home_url") or site_config.get("discovery_settings", {}).get("home_url")
        if not target_url:
            logger.error(f"[run_site] No target URL specified and no home_url in site_config for '{site_name}'")
            return 1
        logger.info(f"[run_site] Using home_url from site_config: {target_url}")

    # Dry-run mode: validate config and setup only
    if args.dry_run:
        logger.info("[run_site] DRY-RUN mode: Validating configuration and setup...")

        # Check Stealth module
        try:
            from scraping.stealth import apply_stealth_to_context, build_stealth_params_from_site_config  # noqa: F401

            logger.info("[run_site] ✅ Stealth module is available")
            # Test build_stealth_params_from_site_config
            try:
                test_params = build_stealth_params_from_site_config(site_config, settings={})
                logger.info(f"[run_site] ✅ Stealth params built successfully: {list(test_params.keys())}")
            except Exception as test_e:
                logger.warning(f"[run_site] ⚠️  Stealth params test failed: {test_e}")
        except ImportError as e:
            logger.warning(f"[run_site] ⚠️  Stealth module not available: {e}")

        # Check SessionManager
        try:
            from app.agents.browser.session_manager import SessionManager  # noqa: F401

            logger.info("[run_site] ✅ SessionManager is available")
        except ImportError as e:
            logger.error(f"[run_site] ❌ SessionManager not available: {e}")
            return 1

        # Check BrowserUseAgent (already imported at top level)
        try:
            # BrowserUseAgent is already imported at module level
            _ = BrowserUseAgent
            logger.info("[run_site] ✅ BrowserUseAgent is available")
        except NameError as e:
            logger.error(f"[run_site] ❌ BrowserUseAgent not available: {e}")
            return 1

        # Check Moncler patch
        if site_name == "MONCLER_OFFICIAL":
            try:
                from app.agents.browser_use_moncler_patch import moncler_plp_recovery  # noqa: F401

                logger.info("[run_site] ✅ Moncler patch is available")
            except ImportError as e:
                logger.warning(f"[run_site] ⚠️  Moncler patch not available: {e}")

        # Validate site_config structure
        required_keys = ["discovery_settings", "selectors"]
        missing_keys = [k for k in required_keys if k not in site_config]
        if missing_keys:
            logger.warning(f"[run_site] ⚠️  Site config missing keys: {missing_keys}")
        else:
            logger.info("[run_site] ✅ Site config structure is valid")

        logger.info("[run_site] ✅ DRY-RUN validation complete. Configuration looks good.")
        logger.info(f"[run_site] Would run with: site={site_name}, query={args.query}, url={target_url}")
        return 0

    # Actual run
    logger.info(f"[run_site] Starting actual run for '{site_name}'...")

    # Override settings from CLI args
    if "discovery_settings" not in site_config:
        site_config["discovery_settings"] = {}
    ds = site_config["discovery_settings"]
    ds["timeout_sec"] = args.timeout
    if args.enable_video:
        ds["enable_video"] = True
    if args.human_like:
        ds["enable_human_like"] = True

    run_ctx = RunContext()

    # CR-E2E-003B拡張: system.logをrunディレクトリに確実に出力する
    try:
        system_log_handler = run_ctx.setup_system_logger()
        root_logger = logging.getLogger()
        root_logger.addHandler(system_log_handler)
        logger.info(f"[run_site] System log will be saved to: {run_ctx.get_path('system.log')}")
    except Exception as e:
        logger.warning(f"[run_site] Failed to setup system logger: {e}", exc_info=True)

    # Proxy mode設定（--proxy-mode優先、--use-proxyは後方互換性のため）
    use_proxy_value = None
    # --proxy-modeが指定されている場合（auto以外）
    if args.proxy_mode != "auto":
        use_proxy_value = args.proxy_mode == "on"
    # --use-proxyが指定されている場合（後方互換性）
    elif hasattr(args, "proxy_mode_on") and args.proxy_mode_on:
        use_proxy_value = True

    runtime_kwargs = {
        "headless": not args.headful,
        "timeout_sec": args.timeout,
        "enable_video": args.enable_video,
        "site": site_name,
        "enable_human_like": args.human_like,
    }

    # use_proxyは明示指定時のみ設定（Noneの場合は設定しない）
    if use_proxy_value is not None:
        runtime_kwargs["use_proxy"] = use_proxy_value

    agent = BrowserUseAgent(
        runtime_kwargs=runtime_kwargs,
        config_provider=config_provider,
    )
    timeout_sec = args.timeout + 60  # buffer for materialize steps

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
        logger.error(f"[run_site] Timeout after {timeout_sec}s; capturing failure snapshot and cancelling task.")
        try:
            await write_fail_snapshot(
                run_ctx,
                getattr(agent, "_page", None),
                getattr(getattr(agent, "_page", None), "url", None),
                TimeoutError(f"Timeout after {timeout_sec}s"),
                site_config,
            )
        except Exception as snap_e:
            logger.error(f"[run_site] write_fail_snapshot failed: {snap_e}")

        # タイムアウト時にもエラーログファイルのパスを表示
        log_file_path = run_ctx.get_path("system.log")
        logger.error(f"[run_site] Error log available at: {log_file_path}")
        logger.error(f"[run_site] Failure snapshot: {run_ctx.run_path}/failure_dom.html")
        logger.error(f"[run_site] Screenshots: {run_ctx.screenshots_path}")
        for t in pending:
            t.cancel()
            with contextlib.suppress(Exception):
                await t
        result = DiscoveryResult(
            ok=False,
            site=site_name,
            query=args.query,
            message=f"Timeout after {timeout_sec}s",
            evidence={"timeout_sec": timeout_sec},
        )
    else:
        result = run_task.result()

    # Save result JSON
    result_dict = {}
    if hasattr(result, "to_dict"):
        result_dict = result.to_dict()
    elif hasattr(result, "__dict__"):
        result_dict = {k: v for k, v in result.__dict__.items() if not k.startswith("_")}
    else:
        result_dict = {
            "ok": getattr(result, "ok", False),
            "site": getattr(result, "site", ""),
            "query": getattr(result, "query", ""),
        }

    # CR-E2E-001: success_stage と criteria を result.json のトップレベルに追加
    # （evidence 内にも存在するが、可読性のためトップレベルにも追加）
    if isinstance(result_dict, dict) and "evidence" in result_dict:
        evidence = result_dict.get("evidence", {})
        if "success_stage" in evidence:
            result_dict["success_stage"] = evidence["success_stage"]
        if "criteria" in evidence:
            result_dict["criteria"] = evidence["criteria"]

    result_path = run_ctx.get_path("result.json")
    result_path.write_text(json.dumps(result_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[run_site] Run finished. result.ok={getattr(result, 'ok', None)} run_dir={run_ctx.run_path}")
    logger.info(f"[run_site] Result saved to: {result_path}")

    # エラーログファイルのパスを表示
    log_file_path = run_ctx.get_path("system.log")
    if log_file_path.exists():
        logger.info(f"[run_site] Error log available at: {log_file_path}")
    else:
        logger.info(f"[run_site] Error log will be saved to: {log_file_path}")

    # 失敗時はエラーログファイルのパスを ERROR レベルで表示
    if not getattr(result, "ok", False):
        logger.error(f"[run_site] Run failed. Check error log: {log_file_path}")
        logger.error(f"[run_site] Failure snapshot: {run_ctx.run_path}/failure_dom.html")
        logger.error(f"[run_site] Screenshots: {run_ctx.screenshots_path}")

    return 0 if getattr(result, "ok", False) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
