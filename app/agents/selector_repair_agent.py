# ==============================================================================
# ファイル名 (File Name): selector_repair_agent.py
# レジストリ (Registry): app/agents/selector_repair_agent.py
# バージョン (Version): 9.1.0J (Phase D-10: Selector Auto-Healing — 分割版)
#
# --- v9.1.0Jでの主な変更点 ---
# - プロンプト構築 → selector_prompt_builder.py に委譲
# - セレクタランキング → selector_ranker.py に委譲
# - レスポンス検証 → selector_validator.py に委譲
# ==============================================================================
from __future__ import annotations

import json
import logging
from typing import Any

from app.agents.browser.selector_prompt_builder import build_selector_repair_prompt
from app.agents.browser.selector_ranker import rank_selectors
from app.agents.browser.selector_validator import extract_json_from_text, normalize_proposal
from app.core.run_context import RunContext

try:
    from app.utils.ai_llm_controller import AILlmController
except ImportError:
    AILlmController = None  # type: ignore

try:
    if AILlmController is None:
        raise ImportError("AILlmController not available")
    _DefaultLLM = AILlmController
except Exception:

    class _DefaultLLM:  # type: ignore
        def generate(self, prompt: str, task_type: str = "default", **kwargs: Any) -> Any:
            logging.info("AiLlmController (Stub): AIへの非同期リクエストをシミュレートします。")
            import time

            time.sleep(0.1)
            dummy_response = {
                "site": "MONCLER_OFFICIAL",
                "page_type": "pdp",
                "strategy": "llm_selector_healing_v1",
                "candidates": [
                    {
                        "target": "title",
                        "old_selector": "h1.product-title",
                        "new_selector": "h1[data-test='product-title']",
                        "confidence": 0.92,
                        "reason": "DOM に data-test='product-title' が追加されているため",
                    }
                ],
            }
            from app.models.result_models import GenerateResult

            return GenerateResult(text=json.dumps(dummy_response, ensure_ascii=False, indent=2))


logger = logging.getLogger(__name__)


class SelectorRepairAgent:
    """CR-ATELIER-003 Phase D-10: Selector Auto-Healing エージェント."""

    def __init__(self, llm_client: Any | None = None):
        if llm_client is None:
            try:
                self.llm_client = _DefaultLLM() if _DefaultLLM else None
            except Exception as e:
                logger.warning(f"[SelectorRepair] Failed to initialize LLM client: {e}, using stub")
                self.llm_client = _DefaultLLM() if _DefaultLLM else None
        else:
            self.llm_client = llm_client

    async def propose_selector_patches(
        self,
        *,
        site: str,
        page_type: str,
        failure_context: dict[str, Any],
        failure_analysis: dict[str, Any],
        dom_snapshot_html: str,
        current_selectors: dict[str, Any],
        site_config: dict[str, Any] | None = None,
        previous_successes: list[dict[str, Any]] | None = None,
        previous_failures: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """LLM を使って新しい CSS セレクタ候補を提案する."""
        try:
            prompt = build_selector_repair_prompt(
                site=site,
                page_type=page_type,
                failure_context=failure_context,
                failure_analysis=failure_analysis,
                dom_snapshot_html=dom_snapshot_html,
                current_selectors=current_selectors,
                site_config=site_config,
                previous_successes=previous_successes or [],
                previous_failures=previous_failures or [],
            )

            logger.info(
                f"[SelectorRepair] Proposing selector patches for {site}/{page_type} "
                f"(error_type={failure_context.get('error_type')})"
            )

            result = await self.llm_client.generate(prompt, task_type="code")

            try:
                proposal = json.loads(result.text)
            except json.JSONDecodeError:
                logger.warning("[SelectorRepair] LLM response is not valid JSON, attempting extraction")
                proposal = extract_json_from_text(result.text)

            normalized = normalize_proposal(proposal, site=site, page_type=page_type)

            candidates = normalized.get("candidates", [])
            if candidates:
                ranked_candidates = rank_selectors(candidates=candidates, site_config=site_config or {})
                normalized["candidates"] = ranked_candidates
                logger.info(
                    f"[SelectorRepair] Ranked {len(ranked_candidates)} selector candidates "
                    f"(top: {ranked_candidates[0].get('new_selector') if ranked_candidates else 'N/A'})"
                )

            logger.info(f"[SelectorRepair] Generated {len(normalized.get('candidates', []))} selector candidates")

            return normalized

        except Exception as e:
            logger.error(f"[SelectorRepair] Failed to propose selector patches: {e}", exc_info=True)
            return {
                "site": site,
                "page_type": page_type,
                "strategy": "llm_selector_healing_v1",
                "candidates": [],
                "error": str(e),
            }

    async def propose_fix(
        self,
        *,
        intent: str,
        failed_selectors: list[str],
        html_content: str,
        site: str,
        site_config: dict,
        run_context: RunContext,
    ) -> dict[str, Any]:
        """既存の propose_fix メソッド（後方互換性のため保持）."""
        logger.warning("[SelectorRepair] propose_fix is deprecated, use propose_selector_patches instead")

        failure_context = {
            "error_type": "selector_not_found",
            "error_message": f"Failed selectors: {', '.join(failed_selectors)}",
        }
        failure_analysis = {
            "summary": intent,
            "root_causes": ["Selector may be outdated"],
            "suggested_fixes": ["Find new selector"],
        }
        current_selectors = site_config.get("selectors", {}).get("pdp", {})

        return await self.propose_selector_patches(
            site=site,
            page_type="pdp",
            failure_context=failure_context,
            failure_analysis=failure_analysis,
            dom_snapshot_html=html_content,
            current_selectors=current_selectors,
        )
