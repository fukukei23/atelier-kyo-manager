# ==============================================================================
# ファイル名 (File Name): app/agents/supplier_scout_agent.py
# レジストリ (Registry): app/agents/supplier_scout_agent.py
# 更新日時 (Date & Time JST): 2025年10月07日 01:49:00
# バージョン (Version): 34.0.0J (Dependency Injection)
#
# --- v34.0.0Jでの主な変更点 ---
# - [責務の分離] あなたの卓越した設計に基づき、本エージェントから設定ファイル
#   読み込みの責務を削除。上位のオーケストレーターから設定を注入される
#   (Dependency Injection) 形式へと変更し、モジュール性を飛躍的に向上。
# - [テスト容易性の向上] エージェントがファイルI/Oに依存しなくなったことで、
#   ユニットテストが極めて容易になり、システム全体の信頼性が向上。
#
# --- 既存機能からの維持 ---
# - v33.0.0Jの堅牢な並行処理、高度な可観測性、自己進化ループといった中核機能は
#   完全に維持・統合されています。
# ==============================================================================
# -*- coding: utf-8 -*-

from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse, urlunparse, urlencode, parse_qsl
import asyncio

# --- プロジェクトルートをPythonの検索パスに追加 (堅牢化) ---
try:
    APP_ROOT = Path(__file__).resolve().parents[2]
    if str(APP_ROOT) not in sys.path:
        sys.path.insert(0, str(APP_ROOT))
except (NameError, IndexError):
    if Path.cwd() not in sys.path:
        sys.path.insert(0, str(Path.cwd()))

from app.models.result_models import DiscoveryResult
from app.core.run_context import RunContext
from app.agents.browser_use_agent import BrowserUseAgent
from app.agents.failure_analysis_agent import FailureAnalysisAgent
from app.utils.overrides_store import update_site_selectors, extract_suggested_selectors

logger = logging.getLogger(__name__)


class SupplierScoutAgent:
    """現場総指揮官。作戦計画を策定・記録し、部隊に任務を割り当て、結果を報告し、知識を恒久化する。"""
    def __init__(self, runtime_kwargs: Optional[Dict[str, Any]] = None):
        self.runtime_kwargs = runtime_kwargs or {}

    async def run(self, *, sites_config: Dict[str, Any], sites: List[str], query: str) -> List[DiscoveryResult]:
        """複数のサイトに対して偵察任務を並行して実行する（設定はオーケストレーターから受け取る）"""
        tasks = [self.run_for_single_site(site=site, query=query, site_config=sites_config.get(site) or {})
                 for site in sites]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        final_results: List[DiscoveryResult] = []
        for i, res in enumerate(results):
            site = sites[i]
            if isinstance(res, DiscoveryResult):
                final_results.append(res)
            else:
                # 予期せぬ例外が発生した場合
                logger.error(f"[SCOUT] Unhandled exception for site='{site}': {res}", exc_info=res)
                rc = RunContext() # 新しいコンテキストを生成
                final_results.append(
                    DiscoveryResult(ok=False, site=site, query=query, message=f"Unhandled agent exception: {str(res)}", evidence={"run_id": rc.run_id})
                )
        return final_results

    async def run_for_single_site(self, *, site: str, query: str, site_config: Dict[str, Any]) -> DiscoveryResult:
        """単一のサイトに対してPLP起点の偵察任務を実行する"""
        rc = RunContext()
        logger.info(f"[SCOUT] Starting mission: site='{site}', query='{query}', run_id='{rc.run_id}'")

        try:
            # --- PLP起点のURL生成ロジック ---
            ds = site_config.get("discovery_settings", {}) or {}
            entry = (site_config.get("entrypoints", {}) or {}).get("catalog_url")

            # 1) フォールバックの原案（“二重 en-int”を含むことがある）
            raw_fallback = "https://www.moncler.com/en-int/en-int/women/outerwear/all-down-jackets"
            source = "entrypoint" if entry else ("discovery_settings" if ds.get("catalog_url") else "fallback")
            target_url = entry or ds.get("catalog_url") or raw_fallback

            # 2) フォールバック／設定URLの正規化：/en-int/ を多重→単一へ圧縮
            u = urlparse(target_url)
            path = u.path or "/"
            # 連続する en-int を 1 つに（例: /en-int/en-int/... -> /en-int/...）
            path = path.replace("//", "/")
            while "/en-int/en-int/" in path:
                path = path.replace("/en-int/en-int/", "/en-int/")
            if not path.startswith("/en-int/"):
                # どのロケール前置きも無ければ先頭に en-int を付与
                path = "/en-int" + (path if path.startswith("/") else f"/{path}")

            # 3) 強制ロケールを常に付与
            q = dict(parse_qsl(u.query))
            q["forceLocale"] = "en-int"
            target_url = urlunparse((u.scheme, u.netloc, path, u.params, urlencode(q), u.fragment))

            # 可観測性: どこから来て何を正規化したか残す
            try:
                rc.save_json("navigation_plan.json", {
                    "strategy": "PLP-First",
                    "source": source,
                    "original_url": u.geturl(),
                    "normalized_path": path,
                    "final_url": target_url,
                    "query": query,
                })
            except Exception as _e:
                logger.debug(f"[SCOUT] Failed to save navigation_plan.json: {_e}")

            # --- BrowserUseAgent の呼び出し ---
            agent = BrowserUseAgent(runtime_kwargs=self.runtime_kwargs)
            logger.info(f"[SCOUT] CALLING BrowserUseAgent.run(site='{site}', query='{query}', "
                        f"target='{target_url}', likely_plp=True)")
            result = await agent.run(
                site=site,
                query=query,
                site_config=site_config,
                run_context=rc,
                target_url=target_url,
                likely_plp=True,
            )

            # --- 自己進化ループ: 発見したセレクタの恒久化 ---
            if result and result.evidence:
                suggested = extract_suggested_selectors(result.evidence)
                if suggested:
                    logger.info(f"[SCOUT] New selectors found for '{site}'. Persisting...: {suggested}")
                    try:
                        # overrides.local.json の絶対（またはプロジェクト相対）パスを渡す
                        overrides_path = Path(__file__).resolve().parents[2] / "app" / "config" / "sites" / "overrides.local.json"
                        update_site_selectors(site=site, new_selectors=suggested, overrides_path=overrides_path)
                        logger.info(f"Successfully updated selectors for '{site}'.")
                    except Exception as update_e:
                        logger.error(f"Failed to persist new selectors for '{site}': {update_e}")
                        result.message = (result.message or "") + " (Selector update failed)"

            ev_keys = list((result.evidence or {}).keys()) if getattr(result, "evidence", None) else []
            logger.info(f"[SCOUT] RESULT: ok={result.ok} msg={result.message} ev={ev_keys}")

            return result

        except Exception as e:
            logger.error(f"[SCOUT] Fatal error on site='{site}', run_id='{rc.run_id}': {e}", exc_info=True)
            analysis_agent = FailureAnalysisAgent(runtime_kwargs=self.runtime_kwargs)
            analysis = analysis_agent.analyze(
                error=e, site=site, site_config=site_config, html_content="", run_context=rc
            )
            return DiscoveryResult(ok=False, site=site, query=query, message=str(e), ai_analysis=analysis, evidence={"run_id": rc.run_id})

