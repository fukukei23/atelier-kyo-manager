# -*- coding: utf-8 -*-
# ==============================================================================
# File: app/agents/proxy_scrape_agent.py
# Date (JST): 2025-11-05 13:05:00
# Version: 1.5.0-apply-integration
#
# 目的:
# - v1.4.0 の自己進化ループ（リトライ、_self_evolve）を維持。
# - [v1.5.0] ★統合: diff の意図を組み込み、CodeUpdateAgent.apply を呼び出す。
# - [v1.5.0] ★統合: SelfHealingAgent の結果 (llm_proposal) と
#   ヒューリスティック (overrides_patch) をマージし、
#   site キーを補完してから CodeUpdateAgent.apply に渡す。
# ==============================================================================
import asyncio
import inspect
import logging
import json
from typing import Any, Dict, Optional # ★ v1.4.0 + diff
# from app.agents.gpt_integration import GPTIntegration # apply が内部で呼ぶため不要

logger = logging.getLogger(__name__)

class ProxyScrapeAgent:
    """
    役割:
      1) orchestrator.run(...) を実行
      2) 失敗 → SelfHealingAgent
      3) 結果 → CodeUpdateAgent.apply (v1.5.0)
         ・[v1.5.0] proposal/patch 監査ログを exports/ に自動保存
         ・overrides.local.json に最小差分を適用（dry_run 可）
         ・コード差分は .diff を出力（自動適用は既定で無効）
      4) 最大 N 回まで指数バックオフ付きで再試行
    """

    def __init__(self):
        pass

    async def run(self, orchestrator: Any, args: Any, site_conf: Dict[str, Any]):
        proxy = site_conf.get("proxy_agent", {}) or {}
        max_cycles = int(proxy.get("max_cycles", 2))
        backoff_base = float(proxy.get("retry_backoff", 2.0))
        dry_run = bool(proxy.get("dry_run", False))

        attempt = 0
        last_exc: Optional[Exception] = None
        while attempt < max_cycles:
            attempt += 1
            try:
                return await self._run_once(orchestrator, args)
            except Exception as e:
                last_exc = e
                logger.warning(f"[Proxy] 実行失敗 #{attempt}: {e}")
                if attempt < max_cycles:
                    failure_context = self._build_failure_context(args, e)
                    await self._self_evolve(
                        failure_context,
                        site_conf,
                        dry_run=dry_run,
                        patch_dir=proxy.get("patch_dir", "instance/patches")
                    )
                    sleep_s = backoff_base ** (attempt - 1)
                    logger.info(f"[Proxy] 再試行まで待機: {sleep_s:.1f}s")
                    await asyncio.sleep(sleep_s)

        raise last_exc or RuntimeError("ProxyScrapeAgent: failed")

    async def _run_once(self, orchestrator: Any, args: Any):
        # ... (v1.4.0 と変更なし) ...
        run = getattr(orchestrator, "run")
        sig = inspect.signature(run)
        wants_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        kw = {
            "sites": getattr(args, "sites", None),
            "site": getattr(args, "sites", [None])[0] if getattr(args, "sites", None) else getattr(args, "site", None),
            "query": getattr(args, "query", None),
            "target_url": getattr(args, "target_url", None),
            "likely_plp": getattr(args, "likely_plp", True),
            "headless": (not getattr(args, "headful", False)),
            "headful": getattr(args, "headful", False),
            "slow_mo": getattr(args, "slow_mo", 0),
            "mode": getattr(args, "mode", "run"),
        }
        kw = {k: v for k, v in kw.items() if v is not None}
        if wants_kwargs:
            return await (run(**kw) if inspect.iscoroutinefunction(run) else self._call_sync(run, **kw))
        accepted = set(inspect.signature(run).parameters.keys())
        slim = {k: v for k, v in kw.items() if k in accepted}
        return await (run(**slim) if inspect.iscoroutinefunction(run) else self._call_sync(run, **slim))

    async def _call_sync(self, fn, **kw):
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(**kw))

    def _build_failure_context(self, args: Any, exc: Exception) -> Dict[str, Any]:
        # ... (v1.4.0 と変更なし) ...
        return {
            "error": str(exc),
            "site_key": getattr(args, "sites", [None])[0] if getattr(args, "sites", None) else getattr(args, "site", None),
            "query": getattr(args, "query", None),
            "target_url": getattr(args, "target_url", None),
        }

    async def _self_evolve(
        self,
        failure_context: Dict[str, Any],
        site_conf: Dict[str, Any],
        *,
        dry_run: bool,
        patch_dir: str
    ) -> None:
        from app.agents.self_healing_agent import SelfHealingAgent
        from app.agents.code_update_agent import CodeUpdateAgent
        from pathlib import Path

        proxy = site_conf.get("proxy_agent", {}) or {}
        allow_code_changes = bool(proxy.get("allow_code_changes", False))
        llm_cfg = (proxy.get("llm") or {}).get("generation_config")
        site_key = failure_context.get("site_key") or site_conf.get("site_key") or "UNKNOWN"

        agent = SelfHealingAgent()
        result = await agent.execute(
            failure_context=failure_context,
            site_config=site_conf,
            allow_code_changes=allow_code_changes,
            llm_generation_config=llm_cfg,
        )

        updater = CodeUpdateAgent(
            patch_dir=patch_dir,
            enable_git_stage=bool(proxy.get("git_stage", False))
        )

        # --- ★ 統合 (v1.5.0): diff の意図に基づき CodeUpdateAgent.apply を呼ぶ ---

        # 1. SelfHealingAgent の結果 (LLM提案 + Heuristic) をマージ
        #    (diff の or {} と site キー補完も適用)
        final_proposal: Dict[str, Any] = result.get("llm_proposal") or {}
        heuristic_patch: Dict[str, Any] = result.get("overrides_patch") or {} # Heuristic

        final_proposal.setdefault("site", site_key)

        # llm_proposal["overrides_patch"] と heuristic_patch をマージ
        # (注: ここは _deep_merge が理想だが、循環参照を避けるため .update で代用)
        llm_ovr = final_proposal.get("overrides_patch") or {}
        llm_ovr.update(heuristic_patch)
        final_proposal["overrides_patch"] = llm_ovr

        # 2. v1.4.0 の個別監査ロジック (save_proposal, write_unified_diff) と
        #    個別適用ロジック (apply_overrides_patch) を、
        #    v1.3.2 で定義した apply (監査+適用) の単一呼び出しに置き換える。
        try:
            res = updater.apply(final_proposal, dry_run=dry_run)
            logger.info(f"[Proxy] CodeUpdateAgent.apply result: {res}")
        except Exception as e:
            logger.error(f"[Proxy] CodeUpdateAgent.apply failed: {e}")

        # ---------------------------------------------------------------------

        # 3) コード変更は diff を残すのみ（v1.4.0 のロジックを維持）
        #    (apply は code_hints を適用しないため、これは別途必要)
        code_hints = (final_proposal or {}).get("code_hints") or []
        if code_hints:
            diff_text = "\n\n".join([h for h in code_hints if isinstance(h, str)])
            if diff_text.strip():
                updater.emit_code_diff(f"code_hints_{site_key}", diff_text)

        # 4) 監査ログ（v1.4.0 のロジックを維持）
        Path(patch_dir, f"last_result_{site_key}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        logger.info("[Proxy] 自己進化ループの1サイクル完了")
