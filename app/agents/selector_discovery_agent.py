# 2025年09月21日 19:36 JST
# ファイル名: selector_discovery_agent.py
# レジストリ: app/agents/selector_discovery_agent.py
# バージョン: 27.0.0J (Final Consolidated)
#
# --- v27.0.0Jでの主な変更点 ---
# - [最終統合] これまでの対話で指摘された全てのバグ修正
#   （非同期await漏れ、コンストラクタ引数、APIキー名）を統合した最終版です。
# - [自己進化] 知的修復によるセレクタ提案を即座に適用し、抽出を再試行する
#   自己進化ループを実装しています。
# - [非同期安全] `run_context.take_screenshot`の呼び出しを`await`に
#   統一し、非同期処理の安全性を完全に保証します。

from __future__ import annotations
import logging
import asyncio
import copy
from typing import Any, Dict, Optional
import sys
from pathlib import Path

# --- プロジェクトルートをPythonの検索パスに追加 ---
APP_ROOT = Path(__file__).resolve().parents[2]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from playwright.async_api import async_playwright, Page, BrowserContext, Error as PlaywrightError

from app.extractors.product_info_extractor import extract_product_info
from app.agents.failure_analysis_agent import FailureAnalysisAgent
from app.agents.self_healing_agent import SelfHealingAgent
from app.models.result_models import DiscoveryResult
from core.run_context import RunContext

logger = logging.getLogger(__name__)

class SelectorDiscoveryAgent:
    """自己進化ループを備えた、最先端の斥候エージェント。"""
    def __init__(self, runtime_kwargs: Optional[Dict[str, Any]] = None):
        self.runtime_kwargs = runtime_kwargs or {}
        self.analysis_agent = FailureAnalysisAgent(runtime_kwargs=runtime_kwargs)
        self.healing_agent = SelfHealingAgent()

    def _resolve_run_settings(self, site_config: Dict[str, Any]) -> Dict[str, Any]:
        """実行時設定を解決する"""
        site_ds = site_config.get("discovery_settings", {})
        final_settings = {
            "timeout_sec": self.runtime_kwargs.get("timeout_sec") or site_ds.get("timeout_sec", 30),
            "headless": self.runtime_kwargs.get("headless", True),
            "slow_mo": self.runtime_kwargs.get("slow_mo", 0),
            "max_self_heal_attempts": site_ds.get("max_self_heal_attempts", 3),
            "extra_http_headers": self.runtime_kwargs.get("extra_http_headers") or site_ds.get("extra_http_headers")
        }
        return final_settings

    async def run(self, *, site: str, query: str, site_config: Dict[str, Any], run_context: RunContext, target_url: str) -> DiscoveryResult:
        settings = self._resolve_run_settings(site_config)
        attempt_count = 0

        async with async_playwright() as p:
            context: Optional[BrowserContext] = None
            page: Optional[Page] = None
            try:
                browser = await p.chromium.launch(headless=settings["headless"], slow_mo=settings["slow_mo"])

                context_options: Dict[str, Any] = {}
                if settings["extra_http_headers"]:
                    context_options["extraHTTPHeaders"] = settings["extra_http_headers"]

                context = await browser.new_context(**context_options)
                page = await context.new_page()
                page.set_default_timeout(settings["timeout_sec"] * 1000)

                await run_context.take_screenshot(page, "10_discovery_agent_start")
                await page.goto(target_url, wait_until="domcontentloaded")

                # --- 自己進化ループ ---
                current_config = copy.deepcopy(site_config)
                for attempt in range(settings["max_self_heal_attempts"]):
                    attempt_count = attempt + 1
                    await run_context.take_screenshot(page, f"30_discovery_attempt_{attempt_count}_start")

                    html_content = await page.content()
                    extracted_data = extract_product_info(html_content, current_config)

                    if extracted_data.get("price"):
                        logger.info("目標発見！任務成功。")
                        return DiscoveryResult(ok=True, site=site, query=query, message="Discovery successful.", evidence={"extracted_data": extracted_data, "final_config": current_config})

                    logger.warning(f"目標発見に失敗。現場指揮官に自己修復を要請します (RunID: {run_context.run_id})")

                    failed_price_selectors = current_config.get("selectors", {}).get("pdp", {}).get("price", [])
                    heal_result = await self.healing_agent.execute(
                        page=page, run_context=run_context, settings=current_config, attempt=attempt_count,
                        failure_context={
                            "intent": "商品情報の価格(price)の発見",
                            "failed_selectors": failed_price_selectors
                        }
                    )

                    if heal_result.get("strategy") == "repair_proposal":
                        logging.info("知的修復の提案を適用し、抽出を再試行します。")
                        proposal = heal_result.get("proposal", {})
                        new_selectors = proposal.get("proposed_selectors")
                        if new_selectors:
                            logger.info(f"新しい価格セレクタを適用します: {new_selectors}")
                            current_config["selectors"]["pdp"]["price"] = new_selectors
                            continue # ループの先頭に戻って、新しい設定で再試行

                    if not heal_result.get("success"):
                        raise RuntimeError(f"自己修復プロセスが失敗しました: {heal_result.get('message')}")

                raise RuntimeError(f"自己修復の最大試行回数 ({settings['max_self_heal_attempts']}) に達しました。")

            except Exception as e:
                logger.error(f"斥候任務で致命的エラー (RunID: {run_context.run_id}): {e}", exc_info=True)
                html_content = ""
                if page and not page.is_closed():
                    html_content = await page.content()
                    await run_context.take_screenshot(page, "99_exception_state")

                run_context.save_json("failure_summary.json", {"error_type": type(e).__name__, "error_message": str(e), "site": site, "query": query, "attempt": attempt_count})

                analysis = self.analysis_agent.analyze(error=e, site=site, site_config=site_config, html_content=html_content, run_context=run_context)
                message = f"{str(e)}"
                return DiscoveryResult(ok=False, site=site, query=query, message=message, ai_analysis=analysis, evidence={"run_id": run_context.run_id})
            finally:
                if context:
                    await context.close()
