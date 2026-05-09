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

import copy
import logging
import sys
from pathlib import Path
from typing import Any

# --- プロジェクトルートをPythonの検索パスに追加 ---
APP_ROOT = Path(__file__).resolve().parents[2]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from playwright.async_api import BrowserContext, Page, async_playwright

from app.agents.failure_analysis_agent import FailureAnalysisAgent
from app.agents.self_healing_agent import SelfHealingAgent
from app.extractors.product_info_extractor import extract_product_info
from app.models.result_models import DiscoveryResult

try:
    from app.core.run_context import RunContext
except Exception:
    from core.run_context import RunContext

logger = logging.getLogger(__name__)


class SelectorDiscoveryAgent:
    """自己進化ループを備えた、最先端の斥候エージェント。"""

    def __init__(self, runtime_kwargs: dict[str, Any] | None = None):
        self.runtime_kwargs = runtime_kwargs or {}
        self.analysis_agent = FailureAnalysisAgent(runtime_kwargs=runtime_kwargs)
        self.healing_agent = SelfHealingAgent()

    def _resolve_run_settings(self, site_config: dict[str, Any]) -> dict[str, Any]:
        """実行時設定を解決する"""
        site_ds = site_config.get("discovery_settings", {})
        final_settings = {
            "timeout_sec": self.runtime_kwargs.get("timeout_sec") or site_ds.get("timeout_sec", 30),
            "headless": self.runtime_kwargs.get("headless", True),
            "slow_mo": self.runtime_kwargs.get("slow_mo", 0),
            "max_self_heal_attempts": site_ds.get("max_self_heal_attempts", 3),
            "extra_http_headers": self.runtime_kwargs.get("extra_http_headers") or site_ds.get("extra_http_headers"),
        }
        return final_settings

    async def run(
        self, *, site: str, query: str, site_config: dict[str, Any], run_context: RunContext, target_url: str
    ) -> DiscoveryResult:
        settings = self._resolve_run_settings(site_config)
        attempt_count = 0

        async with async_playwright() as p:
            context: BrowserContext | None = None
            page: Page | None = None
            try:
                browser = await p.chromium.launch(headless=settings["headless"], slow_mo=settings["slow_mo"])

                context_options: dict[str, Any] = {}
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
                        return DiscoveryResult(
                            ok=True,
                            site=site,
                            query=query,
                            message="Discovery successful.",
                            evidence={"extracted_data": extracted_data, "final_config": current_config},
                        )

                    logger.warning(f"目標発見に失敗。現場指揮官に自己修復を要請します (RunID: {run_context.run_id})")

                    failed_price_selectors = current_config.get("selectors", {}).get("pdp", {}).get("price", [])
                    heal_result = await self.healing_agent.execute(
                        page=page,
                        run_context=run_context,
                        settings=current_config,
                        attempt=attempt_count,
                        failure_context={
                            "intent": "商品情報の価格(price)の発見",
                            "failed_selectors": failed_price_selectors,
                        },
                    )

                    if heal_result.get("strategy") == "repair_proposal":
                        logging.info("知的修復の提案を適用し、抽出を再試行します。")
                        proposal = heal_result.get("proposal", {})
                        new_selectors = proposal.get("proposed_selectors")
                        if new_selectors:
                            logger.info(f"新しい価格セレクタを適用します: {new_selectors}")
                            current_config["selectors"]["pdp"]["price"] = new_selectors
                            continue  # ループの先頭に戻って、新しい設定で再試行

                    if not heal_result.get("success"):
                        raise RuntimeError(f"自己修復プロセスが失敗しました: {heal_result.get('message')}")

                raise RuntimeError(f"自己修復の最大試行回数 ({settings['max_self_heal_attempts']}) に達しました。")

            except Exception as e:
                logger.error(f"斥候任務で致命的エラー (RunID: {run_context.run_id}): {e}", exc_info=True)
                html_content = ""
                if page and not page.is_closed():
                    html_content = await page.content()
                    await run_context.take_screenshot(page, "99_exception_state")

                run_context.save_json(
                    "failure_summary.json",
                    {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "site": site,
                        "query": query,
                        "attempt": attempt_count,
                    },
                )

                analysis = self.analysis_agent.analyze(
                    error=e, site=site, site_config=site_config, html_content=html_content, run_context=run_context
                )
                message = f"{str(e)}"
                return DiscoveryResult(
                    ok=False,
                    site=site,
                    query=query,
                    message=message,
                    ai_analysis=analysis,
                    evidence={"run_id": run_context.run_id},
                )
            finally:
                if context:
                    await context.close()

    async def propose_moncler_selectors(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        CR-ATELIER-002 Step 6-5: Moncler 専用のセレクタ提案

        Args:
            payload: セレクタ発見のための情報
                - dom_snapshot_path: Optional[str]
                - selectors_current: Dict[str, Any]
                - layer_stats: Dict[str, Any]
                - rejection_stats: Dict[str, Any]
                - run_id: Optional[str]

        Returns:
            Dict[str, Any]: セレクタ提案結果
                - candidate_selectors: List[str]
                - confidence_scores: List[float]
                - recommended_layer: str ("primary" | "secondary" | "tertiary")
        """
        logger.info("[SelectorDiscovery][Moncler] Starting selector proposal")

        payload.get("dom_snapshot_path")
        selectors_current = payload.get("selectors_current", {})
        layer_stats = payload.get("layer_stats", {})

        # 現在のセレクタを取得
        plp_selectors = (selectors_current.get("plp", {}) or {}).get("pdp_link_selectors", []) or []

        # 候補セレクタを生成（現時点ではルールベース、将来は LLM を使用可能）
        candidate_selectors = []
        confidence_scores = []

        # Primary layer の候補（既存のセレクタをベースに拡張）
        if layer_stats.get("primary_raw", 0) == 0:
            # Primary が失敗した場合、より広いセレクタを提案
            candidate_selectors.extend(
                [
                    "article[data-component*='ProductCard'] a[href*='/products/']",
                    "[data-testid*='product-card'] a[href*='/products/']",
                    "[data-testid*='product-tile'] a[href*='/products/']",
                    "div[class*='product-card' i] a[href*='/products/']",
                    "div[class*='product-tile' i] a[href*='/products/']",
                ]
            )
            confidence_scores.extend([0.92, 0.88, 0.85, 0.80, 0.75])

        # Secondary layer の候補
        if layer_stats.get("secondary_raw", 0) == 0:
            candidate_selectors.extend(
                [
                    "a[href*='/products/']:not([class*='breadcrumb']):not([class*='nav'])",
                    "section a[href*='/products/']",
                    "main a[href*='/products/']",
                ]
            )
            confidence_scores.extend([0.70, 0.65, 0.60])

        # Tertiary layer の候補（最終手段）
        if layer_stats.get("tertiary_raw", 0) == 0:
            candidate_selectors.extend(
                [
                    "a[href*='/products/']",
                ]
            )
            confidence_scores.extend([0.50])

        # 既存のセレクタと重複を排除
        existing_selectors = set(plp_selectors)
        unique_candidates = []
        unique_scores = []
        for sel, score in zip(candidate_selectors, confidence_scores):
            if sel not in existing_selectors:
                unique_candidates.append(sel)
                unique_scores.append(score)

        # 信頼度の高い順にソート
        sorted_pairs = sorted(zip(unique_candidates, unique_scores), key=lambda x: x[1], reverse=True)
        candidate_selectors = [sel for sel, _ in sorted_pairs]
        confidence_scores = [score for _, score in sorted_pairs]

        # 最低3件の候補を返す（不足する場合は既存のセレクタから追加）
        if len(candidate_selectors) < 3:
            # 既存のセレクタから追加（信頼度は低めに設定）
            for sel in plp_selectors[:3]:
                if sel not in candidate_selectors:
                    candidate_selectors.append(sel)
                    confidence_scores.append(0.40)

        # recommended_layer を決定
        recommended_layer = "primary"
        if layer_stats.get("primary_accepted", 0) == 0:
            if layer_stats.get("secondary_accepted", 0) > 0:
                recommended_layer = "secondary"
            else:
                recommended_layer = "tertiary"

        result = {
            "candidate_selectors": candidate_selectors[:10],  # 最大10件
            "confidence_scores": confidence_scores[:10],
            "recommended_layer": recommended_layer,
        }

        logger.info(
            f"[SelectorDiscovery][Moncler] Proposed {len(result['candidate_selectors'])} selectors, "
            f"recommended_layer={recommended_layer}"
        )

        return result
