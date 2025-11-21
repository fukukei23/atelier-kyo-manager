# ==============================================================================
# ファイル名 (File Name): app/agents/selector_discovery_agent.py
# レジストリ (Registry): app/agents/selector_discovery_agent.py
# 更新日時 (Date & Time JST): 2025年10月03日 16:14:00
# バージョン (Version): 31.0.0J (Resilient Fallback)
#
# --- v31.0.0Jでの主な変更点 ---
# - [安全なフォールバック提案] あなたの提案に基づき、例外処理ブロックで終了する際に、
#   最後の砦として「安全で汎用的なセレクタ候補群」をevidenceに含めて返すように修正。
# - [学習ループの確実性向上] これにより、自己修復プロセスが完全に失敗した場合でも、
#   BrowserUseAgentは最低限の行動指針を得ることができ、自己進化の試みを
#   継続することが可能になります。
#
# --- 既存機能からの維持 ---
# - v30.5.0Jで確立された、API整合性、安全な重複排除、ステルス能力、
#   証跡保全といった全ての機能は完全に維持されています。
# ==============================================================================
# -*- coding: utf-8 -*-

from __future__ import annotations
import logging
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- プロジェクトルートをPythonの検索パスに追加 ---
try:
    APP_ROOT = Path(__file__).resolve().parents[2]
    if str(APP_ROOT) not in sys.path:
        sys.path.insert(0, str(APP_ROOT))
except (NameError, IndexError):
    if Path.cwd() not in sys.path:
        sys.path.insert(0, str(Path.cwd()))

from playwright.async_api import async_playwright, Page, Browser, BrowserContext

from app.core.run_context import RunContext
from app.models.result_models import DiscoveryResult
from app.agents.failure_analysis_agent import FailureAnalysisAgent
from app.agents.self_healing_agent import SelfHealingAgent

logger = logging.getLogger(__name__)


def _hashable_key(sel: Any) -> Any:
    if isinstance(sel, dict):
        return tuple(sorted(sel.items()))
    return sel


class SelectorDiscoveryAgent:
    """リンク0件時に自己修復を統括し、新しいセレクタ候補を提示する斥候エージェント。"""

    def __init__(self, runtime_kwargs: Optional[Dict[str, Any]] = None):
        self.runtime_kwargs = runtime_kwargs or {}
        self.analysis_agent = FailureAnalysisAgent(runtime_kwargs=self.runtime_kwargs)
        self.healing_agent = SelfHealingAgent()

    def _resolve_run_settings(self, site_config: Dict[str, Any]) -> Dict[str, Any]:
        ds = site_config.get("discovery_settings", {}) or {}
        return {
            "timeout_sec": self.runtime_kwargs.get("timeout_sec") or ds.get("timeout_sec", 30),
            "headless": self.runtime_kwargs.get("headless", True),
            "slow_mo": self.runtime_kwargs.get("slow_mo", 0),
            "viewport": ds.get("viewport"),
            "user_agent": ds.get("user_agent"),
            "extra_http_headers": ds.get("extra_http_headers"),
            "enable_har": self.runtime_kwargs.get("enable_har", True),
        }

    async def run(
        self, *,
        site: str, query: str, site_config: Dict[str, Any],
        run_context: RunContext, target_url: str
    ) -> DiscoveryResult:

        settings = self._resolve_run_settings(site_config)

        async with async_playwright() as p:
            browser: Optional[Browser] = None
            context: Optional[BrowserContext] = None
            page: Optional[Page] = None

            try:
                browser = await p.chromium.launch(
                    headless=settings["headless"],
                    slow_mo=settings["slow_mo"]
                )

                ctx_opts: Dict[str, Any] = {}
                if settings.get("viewport"): ctx_opts["viewport"] = settings["viewport"]
                if settings.get("user_agent"): ctx_opts["user_agent"] = settings["user_agent"]
                if settings.get("extra_http_headers"): ctx_opts["extra_http_headers"] = settings["extra_http_headers"]
                if settings.get("enable_har"):
                    ctx_opts["record_har_path"] = str(run_context.get_path("network_discovery.har"))

                context = await browser.new_context(**ctx_opts)
                page = await context.new_page()
                page.set_default_timeout(int(settings["timeout_sec"]) * 1000)

                await page.goto(target_url, wait_until="domcontentloaded")
                await run_context.take_screenshot(page, "30_discovery_agent_start")

                heal_result = await self.healing_agent.execute(
                    page=page,
                    run_context=run_context,
                    settings=site_config,
                    attempt=1,
                    failure_context={
                        "intent": "商品一覧ページ(PLP)から商品詳細ページ(PDP)へのリンクを発見する",
                        "failed_selectors": site_config.get("selectors", {}).get("pdp", {}).get("pdp_link_selectors", [])
                    }
                )

                if heal_result.get("strategy") == "repair_proposal":
                    proposal = heal_result.get("proposal", {}) or {}
                    raw_candidates = proposal.get("proposed_selectors") or []

                    unique_candidates: List[Any] = []
                    seen = set()
                    for cand in raw_candidates:
                        key = _hashable_key(cand)
                        if key not in seen:
                            seen.add(key)
                            unique_candidates.append(cand)

                    if not unique_candidates:
                        raise RuntimeError("自己修復は実行されましたが、新しいセレクタ候補が得られませんでした。")

                    evidence = {
                        "suggested_selectors": {
                            "pdp": {"pdp_link_selectors": unique_candidates}
                        },
                        "ai_analysis": heal_result.get("analysis_log")
                    }

                    return DiscoveryResult(
                        ok=True, site=site, query=query,
                        message="Selector discovery successful.", evidence=evidence
                    )

                raise RuntimeError(f"自己修復プロセスがセレクタを提案できませんでした: {heal_result.get('message')}")

            except Exception as e:
                logger.error(f"セレクタ自動探索任務で致命的エラー (RunID: {run_context.run_id}): {e}", exc_info=True)

                html_content = ""
                if page and not page.is_closed():
                    try:
                        html_content = await page.content()
                        await run_context.take_screenshot(page, "99_discovery_exception_state")
                        run_context.save_text("99_discovery_exception_state.html", html_content)
                    except Exception: pass

                analysis = self.analysis_agent.analyze(
                    error=e, site=site, site_config=site_config,
                    html_content=html_content, run_context=run_context
                )

                # ▼▼▼ v31.0.0J 修正点 ▼▼▼
                # 任務失敗時でも、汎用的なセレクタ候補を「最後の砦」として提案する
                fallback_suggested = {
                    "pdp": {
                        "pdp_link_selectors": [
                            "a[data-qa='product-tile-link']",
                            "a[href*='/products/']",
                            "a[href*='/product/']",
                            "a[href*='/p/']",
                            "a:has(img)"
                        ]
                    }
                }
                # ▲▲▲ v31.0.0J 修正点 ▲▲▲

                return DiscoveryResult(
                    ok=False,
                    site=site,
                    query=query,
                    message=f"SelectorDiscoveryAgent failed: {e}",
                    ai_analysis=analysis,
                    # ▼▼▼ v31.0.0J 修正点 ▼▼▼
                    evidence={"run_id": run_context.run_id, "suggested_selectors": fallback_suggested}
                    # ▲▲▲ v31.0.0J 修正点 ▲▲▲
                )

            finally:
                if context: await context.close()
                if browser: await browser.close()

