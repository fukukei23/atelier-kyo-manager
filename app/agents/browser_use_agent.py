# 2025年09月21日 15:49 JST
# ファイル名: browser_use_agent.py
# レジストリ: app/agents/browser_use_agent.py
# バージョン: 9.0.0J (Hardened & API Compliant)
#
# --- v9.0.0Jでの主な変更点 ---
# - [API互換性] あなたの指摘に基づき、Playwrightの`new_context`が期待する
#   `extraHTTPHeaders`キー（キャメルケース）に正しくマッピングするよう修正。
# - [堅牢化] `enable_har`がFalseの場合に`record_har_path`キー自体を
#   渡さないようにし、None値による潜在的なエラーを回避。
# - [型安全] `tracing.stop`に渡すパスを明示的に`str()`にキャスト。

from __future__ import annotations
import logging
import asyncio
from typing import Any, Dict, Optional
import sys
from pathlib import Path

# --- プロジェクトルートをPythonの検索パスに追加 ---
APP_ROOT = Path(__file__).resolve().parents[2]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from playwright.async_api import (
    async_playwright, Page, BrowserContext, Request,
    Error as PlaywrightError
)

from app.extractors.product_info_extractor import extract_product_info
from app.agents.failure_analysis_agent import FailureAnalysisAgent
from app.models.result_models import DiscoveryResult
from core.run_context import RunContext

logger = logging.getLogger(__name__)

class BrowserUseAgent:
    """
    標準的なブラウザ操作を実行し、高度な観測能力とリソース制御機能を持つエージェント。
    """
    def __init__(self, runtime_kwargs: Optional[Dict[str, Any]] = None):
        self.runtime_kwargs = runtime_kwargs or {}
        self.analysis_agent = FailureAnalysisAgent(runtime_kwargs=runtime_kwargs)

    def _resolve_run_settings(self, site_config: Dict[str, Any]) -> Dict[str, Any]:
        """実行時設定を解決する"""
        site_ds = site_config.get("discovery_settings", {})
        final_settings = {
            "timeout_sec": self.runtime_kwargs.get("timeout_sec") or site_ds.get("timeout_sec", 30),
            "headless": self.runtime_kwargs.get("headless", True),
            "slow_mo": self.runtime_kwargs.get("slow_mo", 0),
            "viewport": site_ds.get("viewport"),
            "user_agent": site_ds.get("user_agent"),
            "extra_http_headers": site_ds.get("extra_http_headers"),
            "enable_trace": site_ds.get("enable_trace", True),
            "enable_har": site_ds.get("enable_har", True),
        }
        return final_settings

    async def run(self, *, site: str, query: str, site_config: Dict[str, Any], run_context: RunContext, target_url: str) -> DiscoveryResult:
        settings = self._resolve_run_settings(site_config)
        timeout_ms = settings["timeout_sec"] * 1000

        logger.info(f"標準任務を開始します (Trace: {'有効' if settings['enable_trace'] else '無効'}, HAR: {'有効' if settings['enable_har'] else '無効'})")

        async with async_playwright() as p:
            context: Optional[BrowserContext] = None
            page: Optional[Page] = None
            try:
                browser = await p.chromium.launch(headless=settings["headless"], slow_mo=settings["slow_mo"])

                # --- ★★★ context_optionsの堅牢化 ★★★ ---
                context_options: Dict[str, Any] = {}
                if settings["viewport"]:
                    context_options["viewport"] = settings["viewport"]
                if settings["user_agent"]:
                    context_options["user_agent"] = settings["user_agent"]
                if settings["extra_http_headers"]:
                    context_options["extraHTTPHeaders"] = settings["extra_http_headers"] # キー名を修正
                if settings["enable_har"]:
                    context_options["record_har_path"] = str(run_context.get_path("network.har"))

                context = await browser.new_context(**context_options)

                if settings["enable_trace"]:
                    await context.tracing.start(screenshots=True, snapshots=True, sources=True)

                page = await context.new_page()
                page.set_default_timeout(timeout_ms)
                run_context.take_screenshot(page, "10_browser_agent_start")

                await page.goto(target_url, wait_until="domcontentloaded")
                run_context.take_screenshot(page, "20_navigation_complete")

                html_content = await page.content()
                extracted_data = extract_product_info(html_content, site_config)
                run_context.save_json("extracted_data.json", extracted_data)

                if not extracted_data.get("price"):
                    raise ValueError("価格が抽出できませんでした。")

                return DiscoveryResult(ok=True, site=site, query=query, message="Standard operation successful.")

            except Exception as e:
                logger.error(f"標準任務で致命的エラー (RunID: {run_context.run_id}): {e}", exc_info=True)
                html_content = ""
                if page and not page.is_closed():
                    run_context.take_screenshot(page, "99_exception_state")
                    html_content = await page.content()

                run_context.save_json("failure_summary.json", {"error_type": type(e).__name__, "error_message": str(e), "site": site, "query": query, "last_known_url": page.url if page else "N/A"})
                analysis_result = self.analysis_agent.analyze(error=e, site=site, site_config=site_config, html_content=html_content, run_context=run_context)

                error_message = f"{str(e)} (詳細はRunID: {run_context.run_id} を参照)"
                return DiscoveryResult(ok=False, site=site, query=query, message=error_message, ai_analysis=analysis_result)
            finally:
                if context:
                    if settings["enable_trace"]:
                        # 重要な注意: tracing.stopはcontext.closeの前に呼び出す必要があります
                        await context.tracing.stop(path=str(run_context.get_path("trace.zip")))
                    await context.close()
