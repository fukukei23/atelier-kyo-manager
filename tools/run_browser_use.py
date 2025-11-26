#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal runner to invoke BrowserUseAgent directly for a single site/URL.
Uses saved session (instance/sessions/<site>.json) if present.

Example:
  .venv/Scripts/python.exe tools/run_browser_use.py \
    --site MONCLER_OFFICIAL \
    --url "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB" \
    --query "down jacket" \
    --headful
"""
from __future__ import annotations
import argparse
import asyncio
import logging
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents.browser_use_agent import BrowserUseAgent
from app.core.run_context import RunContext
from app.models.result_models import DiscoveryResult
from app.utils.observability import write_fail_snapshot


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Direct runner for BrowserUseAgent (single site/URL).")
    p.add_argument("--site", required=True, help="Site key (e.g., MONCLER_OFFICIAL)")
    p.add_argument("--url", required=True, help="Target URL to open (PLP/PDP).")
    p.add_argument("--query", default="", help="Query/brand label (for logging).")
    p.add_argument("--headful", action="store_true", help="Run with headful browser (default: headless).")
    p.add_argument("--enable-video", action="store_true", help="Enable Playwright video recording.")
    p.add_argument("--timeout", type=int, default=60, help="Timeout seconds (default: 60).")
    p.add_argument("--use-proxy", action="store_true", help="Enable proxy usage if configured.")
    p.add_argument("--human-like", action="store_true", help="Enable human-like cursor/scroll pauses.")
    return p.parse_args()


async def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

    # Minimal site_config; discovery_settings can be expanded if needed.
    site_config = {
        "id": args.site,
        "name": args.site,
        "discovery_settings": {
            "timeout_sec": args.timeout,
            "enable_video": args.enable_video,
            "enable_har": False,
            "enable_trace": False,
            "enable_human_like": args.human_like,
        },
        "selectors": {},
    }

    run_ctx = RunContext()
    runtime_kwargs = {
        "headless": not args.headful,
        "timeout_sec": args.timeout,
        "enable_video": args.enable_video,
        "use_proxy": args.use_proxy,
        "site": args.site,
        "enable_human_like": args.human_like,
    }

    agent = BrowserUseAgent(runtime_kwargs=runtime_kwargs)
    # Run with manual timeout supervision to allow failure snapshot on timeout
    timeout_sec = args.timeout + 60  # buffer for materialize steps
    run_task = asyncio.create_task(
        agent.run(
            site=args.site,
            query=args.query,
            site_config=site_config,
            run_context=run_ctx,
            target_url=args.url,
            likely_plp=True,
        )
    )
    done, pending = await asyncio.wait({run_task}, timeout=timeout_sec)
    if not done:
        logging.error(f"[runner] Timeout after {timeout_sec}s; capturing failure snapshot and cancelling task.")
        # Failure snapshot
        try:
            await write_fail_snapshot(
                run_ctx,
                getattr(agent, "_page", None),
                getattr(getattr(agent, "_page", None), "url", None),
                TimeoutError(f"Timeout after {timeout_sec}s"),
                site_config,
            )
        except Exception as snap_e:
            logging.error(f"[runner] write_fail_snapshot failed: {snap_e}")
        # Cancel underlying task
        for t in pending:
            t.cancel()
            try:
                await t
            except Exception:
                pass
        # Build failure result
        result = DiscoveryResult(
            ok=False,
            site=args.site,
            query=args.query,
            message=f"Timeout after {timeout_sec}s",
            evidence={"timeout_sec": timeout_sec},
        )
    else:
        result = run_task.result()

    # Save result JSON to the run directory for inspection
    (run_ctx.get_path("result.json")).write_text(result.json(exclude_none=True, indent=2), encoding="utf-8")
    print(f"[runner] Run finished. result.ok={getattr(result,'ok',None)} run_dir={run_ctx.run_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
