# -*- coding: utf-8 -*-
# ==============================================================================
# File Name : app/utils/observability.py
# Version   : 3.2.2J (Logger Fallback + Maybe-Await, aligned with RunContext 1.6)
# Date (JST): 2025-10-20 00:20
#
# 変更点要旨
# - (V3.2.2) RunContext v1.6 のシグネチャ `save_content(filename, data)` に追従。
# - (V3.2.1) 堅牢化:
#   1. `_log` ヘルパを追加し、`run_context.logger` が存在しない場合に
#      モジュールロガーへ安全にフォールバックする機構を導入。
#   2. `_maybe_await` ヘルパを追加し、`run_context`の保存メソッドが
#      同期的・非同期どちらで実装されていても動作を保証。
#
# 使用方法
# - このモジュールは、`BrowserUseAgent` 等から呼び出されることを想定しています。
#   多様な `RunContext` の実装に対して、安定した観測機能を提供します。
# ==============================================================================
from __future__ import annotations
import time
import json
import logging
import inspect
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.async_api import Page
    from app.core.run_context import RunContext

def _log(rc: "RunContext"):
    """run_context.logger が無ければモジュールロガーにフォールバック"""
    return getattr(rc, "logger", logging.getLogger("observability"))

async def _maybe_await(x):
    """引数が Awaitable であれば await し、そうでなければそのまま返す"""
    return (await x) if inspect.isawaitable(x) else x

async def save_dom(run_context: "RunContext", page: "Page", name: str):
    """Saves the current DOM of the page as an HTML file."""
    if not page or page.is_closed():
        return
    try:
        html = await page.content()
        await _maybe_await(run_context.save_content(f"{name}.html", html))
    except Exception as e:
        _log(run_context).warning(f"Failed to save DOM for '{name}': {e}")

async def save_json(run_context: "RunContext", data: Any, name: str):
    """Saves any data as a JSON file."""
    try:
        await _maybe_await(run_context.save_json(f"{name}.json", data))
    except Exception as e:
        _log(run_context).warning(f"Failed to save JSON for '{name}': {e}")

async def count_selectors(run_context: "RunContext", page: "Page", selectors: List[str], *, name: str = "selector_counts"):
    """Counts elements for a list of selectors and saves the result as JSON."""
    if not page or page.is_closed():
        return
    counts: Dict[str, int] = {}
    for s in selectors or []:
        try:
            counts[s] = await page.locator(s).count()
        except Exception:
            counts[s] = -1
    await save_json(run_context, {"selector_counts": counts}, name)

async def save_raw_hrefs(run_context: "RunContext", hrefs: List[str], *, name: str = "raw_hrefs", limit: int = 200):
    """Saves a list of URLs as a JSON file."""
    try:
        data = {"count": len(hrefs or []), "raw_hrefs": list(hrefs or [])[:limit]}
        await save_json(run_context, data, name)
    except Exception as e:
        _log(run_context).warning(f"Failed to save raw hrefs '{name}': {e}")

async def write_fail_snapshot(run_context: "RunContext", page: Optional["Page"], final_url: Optional[str], error: Exception, site_config: Dict):
    """Saves a snapshot of failure information (DOM, screenshot, etc.)."""
    visible_counts: Dict[str, int] = {}
    note = f"Final URL: {final_url}\nError: {error}"
    screenshot_path: Optional[str] = None

    if page and not page.is_closed():
        await save_dom(run_context, page, "failure_dom")
        try:
            # Best-effort screenshot on failure
            screenshot_path = await run_context.take_screenshot(page, "99_failure")
        except Exception as e:
            _log(run_context).warning(f"Failed to take failure screenshot: {e}")

        try:
            pdp_cfg = (site_config.get("selectors") or {}).get("pdp", {}) or {}
            markers = list(dict.fromkeys((pdp_cfg.get("title_selectors") or []) + (pdp_cfg.get("price_selectors") or [])))
            for s in markers:
                try:
                    visible_counts[s] = await page.locator(s).count()
                except Exception:
                    visible_counts[s] = -1
        except Exception as e:
            _log(run_context).warning(f"Could not count visible markers: {e}")

    md_report = f"""# Failure Snapshot

- **Run ID:** `{getattr(run_context, 'run_id', 'unknown')}`
- **Timestamp:** `{time.strftime('%Y-%m-%d %H:%M:%S Z', time.gmtime())}`
- **Note:** `{note}`

## Key Selector Counts at Failure
```json
{json.dumps(visible_counts, indent=2)}
```

## Available Artifacts
- `failure_dom.html` (if page was available)
- `{screenshot_path or '99_failure.png (if captured)'}`
- `plp_dom_initial_materialized.html`
- `plp_dom_search_fallback.html`
- `selector_counts_plp_initial.json`
- `selector_counts_after_search_fallback.json`
- `raw_hrefs_raw_abs.json`
- `network.har`
- `trace.zip`
"""
    await _maybe_await(run_context.save_content("fail_snapshot.md", md_report))


async def log_operation_metric(
    run_context: "RunContext",
    operation: str,
    duration_ms: float,
    success: bool,
    details: Optional[Dict[str, Any]] = None
):
    """
    操作メトリクスをJSONログファイルに記録する。

    使用例:
        start = time.time()
        # 何らかの操作
        await log_operation_metric(ctx, "page_navigate", (time.time() - start) * 1000, True)
    """
    metric = {
        "operation": operation,
        "duration_ms": round(duration_ms, 2),
        "success": success,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S Z', time.gmtime()),
        "run_id": getattr(run_context, 'run_id', 'unknown'),
    }
    if details:
        metric["details"] = details

    _log(run_context).info(f"[METRIC] {operation}: {duration_ms:.0f}ms {'OK' if success else 'FAIL'}")
    await save_json(run_context, metric, f"metric_{operation}_{int(time.time())}")
