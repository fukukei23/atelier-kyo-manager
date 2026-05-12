# 2025年09月21日 15:59 JST
# ファイル名: supplier_scout_agent.py
# レジストリ: app/agents/supplier_scout_agent.py
# バージョン: 21.0.0J (Evidential Reporting)
#
# --- v21.0.0Jでの主な変更点 ---
# - [報告プロトコルの統一] あなたの指摘に基づき、生成する全ての
#   DiscoveryResultに、`evidence`フィールド経由で`run_id`を
#   含めるようにしました。これにより、司令部での最終集計が
#   エラーなく行えるようになります。

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from app.agents.browser_use_agent import BrowserUseAgent
from app.agents.healing.failure_analysis_agent import FailureAnalysisAgent
from app.agents.selector_discovery_agent import SelectorDiscoveryAgent
from app.core.run_context import RunContext
from app.models.result_models import DiscoveryResult

logger = logging.getLogger(__name__)


class SupplierScoutAgent:
    """現場総指揮官。作戦計画を策定・記録し、部隊に任務を割り当て、結果を報告する。"""

    def __init__(self, runtime_kwargs: dict[str, Any]):
        self.runtime_kwargs = runtime_kwargs
        self.analysis_agent = FailureAnalysisAgent(runtime_kwargs=runtime_kwargs)

    def _generate_candidate_urls(self, query: str, site_config: dict, max_urls: int) -> list[str]:
        brand_slug = query.split()[0].lower().replace(" ", "-")
        base_url = site_config.get("home_url", "")
        ds = site_config.get("discovery_settings", {})
        candidate_urls = []
        for tactic in ds.get("navigation_tactics", ["direct_brand_url"]):
            template = ds.get("url_templates", {}).get(tactic)
            if base_url and template:
                url = template.format(brand=brand_slug, locale="ja-jp", gender="men")
                full_url = urljoin(base_url, url)
                candidate_urls.append(full_url)
        return candidate_urls[:max_urls]

    async def run(self, *, sites_config: dict[str, Any], sites: list[str], query: str) -> list[Any]:
        results: list[Any] = []
        discover_selectors = self.runtime_kwargs.get("discover_selectors", False)

        for site in sites:
            run_context = RunContext()
            logging.info(f"--- 斥候任務開始: '{site}' | Query: '{query}' | RunID: {run_context.run_id} ---")

            site_cfg = sites_config.get(site)
            if not site_cfg:
                result = DiscoveryResult(ok=False, site=site, query=query, message="Site config missing.")
                results.append(result)
                continue

            try:
                candidate_urls = self._generate_candidate_urls(query, site_cfg, max_urls=2)
                run_context.save_json("navigation_plan.json", {"candidate_urls": candidate_urls})
                if not candidate_urls:
                    raise ValueError("斥候任務のための目標URLが生成できませんでした。")
                target_url = candidate_urls[0]

                agent_kwargs = {
                    "site": site,
                    "query": query,
                    "site_config": site_cfg,
                    "run_context": run_context,
                    "target_url": target_url,
                }

                if discover_selectors:
                    agent = SelectorDiscoveryAgent(runtime_kwargs=self.runtime_kwargs)
                    result = await agent.run(**agent_kwargs)
                else:
                    agent = BrowserUseAgent(runtime_kwargs=self.runtime_kwargs)
                    result = await agent.run(**agent_kwargs)

                # --- ★★★ 報告書に管理番号(run_id)を付与 ★★★ ---
                if result.evidence is None:
                    result.evidence = {}
                result.evidence["run_id"] = run_context.run_id
                results.append(result)

            except Exception as e:
                logging.error(
                    f"[{site}] 現場総指揮官レベルで致命的なエラー発生 (RunID: {run_context.run_id}): {e}", exc_info=True
                )
                analysis = self.analysis_agent.analyze(
                    error=e, site=site, site_config=site_cfg, html_content="", run_context=run_context
                )

                error_message = f"{str(e)} (詳細はRunID: {run_context.run_id} を参照)"
                # --- ★★★ 失敗報告書にも管理番号(run_id)を付与 ★★★ ---
                result = DiscoveryResult(ok=False, site=site, query=query, message=error_message, ai_analysis=analysis)
                result.evidence = {"run_id": run_context.run_id}
                results.append(result)

        return results
