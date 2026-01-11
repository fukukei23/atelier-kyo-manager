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
import json
import logging
from dataclasses import asdict
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents.browser_use_agent import BrowserUseAgent
from app.core.run_context import RunContext
from app.models.result_models import DiscoveryResult
from app.utils.observability import write_fail_snapshot

# Stage 4: site_configを正しく読み込む
try:
    from app.config.loader import load_full_config, get_site_config
except ImportError:
    load_full_config = None
    get_site_config = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Direct runner for BrowserUseAgent (single site/URL).")
    p.add_argument("--site", required=True, help="Site key (e.g., MONCLER_OFFICIAL)")
    p.add_argument("--url", required=True, help="Target URL to open (PLP/PDP).")
    p.add_argument("--query", default="", help="Query/brand label (for logging).")
    head = p.add_mutually_exclusive_group()
    head.add_argument("--headless", dest="headful", action="store_false", help="Run with headless browser (default).")
    head.add_argument("--headful", dest="headful", action="store_true", help="Run with headful browser.")
    p.set_defaults(headful=False)  # デフォルトはヘッドレス
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

    # Stage 4: site_configを正しく読み込む（MONCLER_OFFICIAL等の設定を使用）
    if get_site_config and load_full_config:
        try:
            site_config = get_site_config(args.site)
            if not site_config:
                logging.warning(f"[runner] Site config not found for {args.site}, using minimal config.")
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
            else:
                # コマンドライン引数で上書き
                if "discovery_settings" not in site_config:
                    site_config["discovery_settings"] = {}
                site_config["discovery_settings"]["timeout_sec"] = args.timeout
                if args.enable_video:
                    site_config["discovery_settings"]["enable_video"] = True
                if args.human_like:
                    site_config["discovery_settings"]["enable_human_like"] = True
                logging.info(f"[runner] Loaded site_config for {args.site}")
        except Exception as e:
            logging.warning(f"[runner] Failed to load site_config: {e}, using minimal config.")
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
    else:
        # フォールバック: 最小限の設定
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
    # Stage 4: DiscoveryResult.to_dict()を使用してJSONに変換
    if hasattr(result, 'to_dict'):
        result_dict = result.to_dict()
    elif hasattr(result, '__dict__'):
        result_dict = asdict(result)
    else:
        result_dict = {"ok": getattr(result, 'ok', False), "site": getattr(result, 'site', ''), "query": getattr(result, 'query', '')}
    (run_ctx.get_path("result.json")).write_text(json.dumps(result_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[runner] Run finished. result.ok={getattr(result,'ok',None)} run_dir={run_ctx.run_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
