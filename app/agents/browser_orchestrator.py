# -*- coding: utf-8 -*-
"""
BrowserOrchestrator - BrowserUseAgent の PLP/PDP フローを統括するオーケストレータ

CR-ATELIER-003 Phase C-3: BrowserUseAgent の PLP/PDP フローを
専任 Orchestrator クラスに切り出す準備として、スケルトンを実装。

現時点では skeleton のみ。将来的に BrowserUseAgent._run_plp_flow と
BrowserUseAgent._run_pdp_flow の中核を置き換える。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable, List

from playwright.async_api import Page, BrowserContext

from app.core.run_context import RunContext
from app.models.result_models import DiscoveryResult
from app.agents.browser.navigation_driver import (
    NavigationDriver,
    NavigationOutcome,
    NavigationContext,
    TrapPageDetected,
)
# BrowserUseAgent と同じ import パスを使用
from app.agents.browser.plp_driver import PlpDriver, PlpNavigationResult
from app.agents.browser.extractor import BrowserExtractionService

# CR-E2E-001: E2E成功段階計算
try:
    from app.utils.e2e_success_stage import compute_success_stage, collect_run_artifacts
except ImportError:
    compute_success_stage = None  # type: ignore
    collect_run_artifacts = None  # type: ignore

# Telemetry のインポート
try:
    from app.agents.browser.telemetry import TelemetryClient
except ImportError:
    TelemetryClient = None  # type: ignore

# SelectorDiscoveryAgent のインポート
try:
    from app.agents.selector_discovery_agent import SelectorDiscoveryAgent
except ImportError:
    SelectorDiscoveryAgent = None  # type: ignore

# FailureAnalysisAgent のインポート
try:
    from app.agents.failure_analysis_agent import FailureAnalysisAgent
except ImportError:
    FailureAnalysisAgent = None  # type: ignore

# SelfHealingPatchAgent のインポート
try:
    from app.agents.self_healing_patch_agent import SelfHealingPatchAgent
except ImportError:
    SelfHealingPatchAgent = None  # type: ignore

# SelfHealingSandbox のインポート
try:
    from app.agents.self_healing_sandbox import SelfHealingSandbox
except ImportError:
    SelfHealingSandbox = None  # type: ignore

# SelfHealingPolicy のインポート
try:
    from app.agents.self_healing_policy import SelfHealingPolicy
except ImportError:
    SelfHealingPolicy = None  # type: ignore

# SelfHealingPatchApplier のインポート
try:
    from app.agents.self_healing_patch_applier import SelfHealingPatchApplier
except ImportError:
    SelfHealingPatchApplier = None  # type: ignore

# SelectorRepairAgent のインポート（CR-ATELIER-003 Phase D-10）
try:
    from app.agents.selector_repair_agent import SelectorRepairAgent
except ImportError:
    SelectorRepairAgent = None  # type: ignore

logger = logging.getLogger(__name__)


class BrowserOrchestrator:
    """
    BrowserUseAgent と NavigationDriver/PlpDriver/SelectorDiscoveryAgent の間に立ち、
    PLP→PDP フロー全体の状態遷移とエラー処理を一元管理するオーケストレータ。

    現時点では skeleton のみ。
    """

    def __init__(
        self,
        *,
        runtime_kwargs: Optional[Dict[str, Any]] = None,
        analysis_agent: Optional[Any] = None,
        discovery_agent: Optional[SelectorDiscoveryAgent] = None,
        patch_agent: Optional[Any] = None,
        sandbox: Optional[Any] = None,
        policy: Optional[Any] = None,
        patch_applier: Optional[Any] = None,
        selector_repair_agent: Optional[Any] = None,
        llm_client: Optional[Any] = None,
        log: Optional[logging.Logger] = None,
    ) -> None:
        """
        BrowserOrchestrator を初期化する。

        Args:
            runtime_kwargs: ランタイム設定
            analysis_agent: FailureAnalysisAgent インスタンス（オプション）     
            discovery_agent: SelectorDiscoveryAgent インスタンス（オプション）  
            patch_agent: SelfHealingPatchAgent インスタンス（オプション）       
            sandbox: SelfHealingSandbox インスタンス（オプション）
            policy: SelfHealingPolicy インスタンス（オプション）
            patch_applier: SelfHealingPatchApplier インスタンス（オプション）
            selector_repair_agent: SelectorRepairAgent インスタンス（オプション、CR-ATELIER-003 Phase D-10）
            llm_client: LLM クライアント（オプション、SelectorRepairAgent 用）
            log: ロガー（オプション）
        """
        self.runtime_kwargs: Dict[str, Any] = runtime_kwargs or {}
        self.log = log or logger

        # FailureAnalysisAgent の初期化
        if analysis_agent is not None:
            self.analysis_agent = analysis_agent
        elif FailureAnalysisAgent is not None:
            self.analysis_agent = FailureAnalysisAgent(runtime_kwargs=self.runtime_kwargs)
        else:
            self.analysis_agent = None

        # SelectorDiscoveryAgent の初期化
        if discovery_agent is not None:
            self.discovery_agent = discovery_agent
        elif SelectorDiscoveryAgent is not None:
            self.discovery_agent = SelectorDiscoveryAgent(runtime_kwargs=self.runtime_kwargs)
        else:
            self.discovery_agent = None
        
        # CR-ATELIER-003 Phase D-6: SelfHealingPatchAgent の初期化
        if patch_agent is not None:
            self.patch_agent = patch_agent
        elif SelfHealingPatchAgent is not None:
            self.patch_agent = SelfHealingPatchAgent(runtime_kwargs=self.runtime_kwargs)
        else:
            self.patch_agent = None
        
        # CR-ATELIER-003 Phase D-8: SelfHealingSandbox の初期化
        if sandbox is not None:
            self.sandbox = sandbox
        elif SelfHealingSandbox is not None:
            self.sandbox = SelfHealingSandbox()
        else:
            self.sandbox = None

        # CR-ATELIER-003 Phase D-9: SelfHealingPolicy の初期化
        if policy is not None:
            self.policy = policy
        elif SelfHealingPolicy is not None:
            policy_path = Path("app/config/self_healing_policy.json")
            self.policy = SelfHealingPolicy.from_file(policy_path)
        else:
            self.policy = None

        # CR-ATELIER-003 Phase D-9: SelfHealingPatchApplier の初期化
        if patch_applier is not None:
            self.patch_applier = patch_applier
        elif SelfHealingPatchApplier is not None:
            self.patch_applier = SelfHealingPatchApplier()
        else:
            self.patch_applier = None

        # CR-ATELIER-003 Phase D-10: SelectorRepairAgent の初期化
        if selector_repair_agent is not None:
            self.selector_repair_agent = selector_repair_agent
        elif SelectorRepairAgent is not None:
            self.selector_repair_agent = SelectorRepairAgent(llm_client=llm_client)
        else:
            self.selector_repair_agent = None

        # CR-ATELIER-003 Phase D-9: overrides.local.json のパス
        self._overrides_path = Path("app/config/sites/overrides.local.json")

    async def run_plp_to_pdp(
        self,
        *,
        page: Page,
        context: BrowserContext,
        site: str,
        query: str,
        site_config: Dict[str, Any],
        settings: Dict[str, Any],
        run_context: RunContext,
        target_url: str,
        start_t: float,
        budget_ms: int,
        nav_outcome: Optional[NavigationOutcome] = None,
        trap_checker: Optional[Callable[[str], bool]] = None,
        telemetry: Optional[Any] = None,
        plugin: Optional[Any] = None,
    ) -> Union[PlpNavigationResult, DiscoveryResult]:
        """
        PLP エントリ→PLP materialize→PDP URL 決定までを統括する。
        - 正常系: PlpNavigationResult を返す
        - 回復不能な失敗: DiscoveryResult(ok=False, ...) を返す

        CR-ATELIER-003 Phase C-4: 最小単位のロジックを移行
        - NavigationDriver.run_plp_flow の呼び出し
        - NavigationOutcome の early return（trap_detected の場合）
        - pdp_links が空のときに PlpDriver.navigate_to_pdp を呼ぶ部分

        Args:
            page: Playwright Page オブジェクト
            context: Playwright BrowserContext オブジェクト
            site: サイトコード
            query: 検索クエリ
            site_config: サイト設定
            settings: 実行設定
            run_context: RunContext オブジェクト
            target_url: ターゲットURL
            start_t: 開始時刻
            budget_ms: 予算時間（ミリ秒）
            nav_outcome: NavigationDriver の実行結果（オプション）
            trap_checker: Trap 判定関数（オプション）
            telemetry: TelemetryClient（オプション）
            plugin: StrategyPlugin（オプション）

        Returns:
            PlpNavigationResult または DiscoveryResult
        """
        self.log.info(
            f"[Debug][Telemetry] telemetry value={telemetry}, type={type(telemetry)}"
        )
        
        # CR-ATELIER-003 Phase C-4: 最小単位のロジック移行
        # Step 1: NavigationContext を構築
        nav_ctx = NavigationContext(
            site=site,
            query=query,
            site_config=site_config,
            settings=settings,
            run_context=run_context,
            start_t=start_t,
            budget_ms=budget_ms,
            entry_url=target_url,
            context=context,
        )

        # Step 2: NavigationDriver を初期化
        if telemetry is None:
            if TelemetryClient is not None:
                telemetry = TelemetryClient(run_context=run_context)
            else:
                telemetry = None

        navigation_driver = NavigationDriver(
            page=page,
            trap_checker=trap_checker,
            telemetry=telemetry,
            strategy=plugin,
        )

        # Step 3: NavigationDriver.run_plp_flow の呼び出し
        # CR-ATELIER-003 Phase D-4: Telemetry 記録を追加
        try:
            if nav_outcome is None:
                # PLP フロー実行前の Telemetry 記録
                if telemetry and hasattr(telemetry, 'record_plp_state'):
                    try:
                        await telemetry.record_plp_state(
                            page,
                            name="plp_dom_initial",
                            site_config=site_config,
                        )
                    except Exception as te:
                        self.log.warning(f"[Orchestrator] Failed to record PLP initial state: {te}", exc_info=True)
                
                nav_outcome = await navigation_driver.run_plp_flow(nav_ctx)
                self.log.debug(f"[Orchestrator] NavigationDriver.run_plp_flow called: entry_url={nav_outcome.entry_url}")
                
                # PLP フロー実行後の Telemetry 記録
                if telemetry and hasattr(telemetry, 'save_json'):
                    try:
                        from app.agents.browser.telemetry import TelemetryContext
                        tctx = TelemetryContext(site=site, query=query, run_id=run_context.run_id, stage="plp_after_nav")
                        await telemetry.save_json(
                            "plp_navigation_outcome",
                            {
                                "entry_url": nav_outcome.entry_url,
                                "pdp_links_count": len(nav_outcome.pdp_links) if nav_outcome else 0,
                                "trap_detected": nav_outcome.trap_detected if nav_outcome else False,
                                "recovered": nav_outcome.recovered if nav_outcome else False,
                            },
                            tctx,
                        )
                    except Exception as te:
                        self.log.warning(f"[Orchestrator] Failed to record PLP navigation outcome: {te}", exc_info=True)
        except TrapPageDetected as trap_e:
            # CR-ATELIER-002 Step 1: Trap ページ検出例外を処理
            self.log.warning(
                f"[Orchestrator] Trap page detected by NavigationDriver: "
                f"type={trap_e.trap_type}, reason={trap_e.reason}, URL={trap_e.url}"
            )
            # CR-ATELIER-003 Phase D-4: Telemetry に trap 検出を記録
            if telemetry:
                try:
                    failure_ctx = self._build_failure_context(
                        site=site,
                        query=query,
                        final_url=trap_e.url,
                        error_type="trap_detected",
                        error_class=type(trap_e).__name__,
                        error_message=str(trap_e),
                        site_config=site_config,
                        run_context=run_context,
                    )
                    # CR-ATELIER-003 Phase D-5: TrapPageDetected の場合は Telemetry に記録するが、
                    # 例外を再スローするため failure_analysis は evidence に含めない
                    await self._maybe_analyze_failure(failure_ctx, page=page, run_context=run_context)
                except Exception as te:
                    self.log.warning(f"[Orchestrator] Failed to record trap detection: {te}", exc_info=True)
            # TrapPageDetected 例外をそのまま再スロー
            raise
        except Exception as nav_e:
            self.log.debug(f"[Orchestrator] NavigationDriver.run_plp_flow failed: {nav_e}")
            nav_outcome = None
            # CR-ATELIER-003 Phase D-4: Telemetry に失敗を記録
            if telemetry:
                try:
                    failure_ctx = self._build_failure_context(
                        site=site,
                        query=query,
                        final_url=page.url,
                        error_type="navigation_failed",
                        error_class=type(nav_e).__name__,
                        error_message=str(nav_e),
                        site_config=site_config,
                        run_context=run_context,
                    )
                        # CR-ATELIER-003 Phase D-5: NavigationDriver 失敗の場合は Telemetry に記録するが、
                    # nav_outcome が None の場合は後続処理で DiscoveryResult を返すため、ここでは analysis を保存しない
                    await self._maybe_analyze_failure(failure_ctx, page=page, run_context=run_context)
                except Exception as te:
                    self.log.warning(f"[Orchestrator] Failed to record navigation failure: {te}", exc_info=True)

        # Step 4: NavigationOutcome の early return（trap_detected の場合）
        if nav_outcome and nav_outcome.trap_detected:
            if nav_outcome.recovered:
                self.log.debug("[Orchestrator] NavigationDriver already handled trap detection and recovery")
            else:
                # NavigationDriver が trap を検出したが復旧に失敗した場合
                # CR-ATELIER-003 Phase D-4: failure_context を標準化
                failure_ctx = self._build_failure_context(
                    site=site,
                    query=query,
                    final_url=page.url,
                    error_type="trap_recovery_failed",
                    error_class="TrapPageDetected",
                    error_message=nav_outcome.trap_reason or "Trap page detected but recovery failed",
                    site_config=site_config,
                    run_context=run_context,
                )
                analysis = await self._maybe_analyze_failure(failure_ctx, page=page, run_context=run_context)
                evidence = {"failure_context": failure_ctx}
                if analysis:
                    evidence["failure_analysis"] = analysis
                    # CR-ATELIER-003 Phase D-6: パッチ候補を生成
                    patch = await self._maybe_build_patch_candidate(
                        failure_context=failure_ctx,
                        failure_analysis=analysis,
                        site_config=site_config,
                        run_context=run_context,
                        page=page,
                    )
                    if patch:
                        evidence["self_healing_patch_candidate"] = patch
                return DiscoveryResult(
                    ok=False,
                    site=site,
                    query=query,
                    message=f"Landing page looks like legal/trap (NavigationDriver recovery failed): {nav_outcome.trap_reason}",
                    evidence=evidence,
                )

        # Step 5: pdp_links が空のときに PlpDriver.navigate_to_pdp を呼ぶ部分
        # CR-ATELIER-003 Phase C-4 Step 2: PlpDriver.navigate_to_pdp の呼び出しを実装
        pdp_links = nav_outcome.pdp_links if nav_outcome else []
        
        if not pdp_links:
            # pdp_links が空の場合、PlpDriver を使用してタイルクリック → PDP遷移
            self.log.warning("[Orchestrator] No PDP links found. Clicking first card using PlpDriver...")
            # CR-ATELIER-003 Phase D-4: Telemetry に pdp_links=0 を記録
            if telemetry and hasattr(telemetry, 'save_json'):
                try:
                    from app.agents.browser.telemetry import TelemetryContext
                    tctx = TelemetryContext(site=site, query=query, run_id=run_context.run_id, stage="plp_no_pdp_links")
                    await telemetry.save_json(
                        "plp_no_pdp_links",
                        {
                            "entry_url": target_url,
                            "current_url": page.url,
                            "pdp_links_count": 0,
                        },
                        tctx,
                    )
                except Exception as te:
                    self.log.warning(f"[Orchestrator] Failed to record no PDP links: {te}", exc_info=True)
            try:
                # BrowserUseAgent と同じ import パスを使用
                # from app.agents.browser.plp_driver import PlpDriver は既にファイル上部で import 済み
                
                # TelemetryClient を取得（PlpDriver に渡すため）
                if telemetry is None:
                    if TelemetryClient is not None:
                        telemetry = TelemetryClient(run_context=run_context)
                    else:
                        telemetry = None
                # telemetry は TelemetryClient として使用
                
                # PlpDriver をインスタンス化（BrowserUseAgent と同じパラメータ）
                plp_driver = PlpDriver(
                    page=page,
                    context=context,
                    site_config=site_config,
                    run_context=run_context,
                    logger=self.log,
                    telemetry=telemetry,
                )
                
                # timeout_ms を計算（BrowserUseAgent と同じロジック）
                default_timeout_ms = int(settings.get("timeout_sec", 60)) * 1000
                if budget_ms is not None:
                    timeout_ms = min(budget_ms, default_timeout_ms)
                else:
                    timeout_ms = default_timeout_ms
                
                # PlpDriver.navigate_to_pdp を呼び出す（BrowserUseAgent と同じパラメータ）
                nav_result = await plp_driver.navigate_to_pdp(
                    target_url=target_url,
                    timeout_ms=timeout_ms,
                    start_t=start_t,
                    budget_ms=budget_ms,
                )
                
                # trap_detected が True の場合は早期終了（クリック継続を抑止）
                if nav_result.trap_detected:
                    self.log.warning(
                        f"[Orchestrator] Trap detected in PlpDriver. Early exit: {nav_result.trap_reason}"
                    )
                    # navigation_failed(trap/locale) として DiscoveryResult を返す
                    failure_ctx = self._build_failure_context(
                        site=site,
                        query=query,
                        final_url=page.url,
                        error_type="trap_detected",
                        error_class="PlpDriverTrapDetection",
                        error_message=f"Trap/locale detected: {nav_result.trap_reason}",
                        site_config=site_config,
                        run_context=run_context,
                    )
                    # failure_ctxからDiscoveryResultを作成
                    return DiscoveryResult(
                        ok=False,
                        site=site,
                        query=query,
                        message=failure_ctx.get("error_message", f"Trap/locale detected: {nav_result.trap_reason}"),
                        evidence=failure_ctx,
                    )
                
                # PlpNavigationResult を返す
                return nav_result
                
            except Exception as plp_e:
                self.log.warning(f"[Orchestrator] PlpDriver failed: {plp_e}", exc_info=True)
                # PlpDriver が失敗した場合、DiscoveryResult を返す
                # CR-ATELIER-003 Phase D-4: failure_context を標準化
                failure_ctx = self._build_failure_context(
                    site=site,
                    query=query,
                    final_url=page.url,
                    error_type="plp_driver_failed",
                    error_class=type(plp_e).__name__,
                    error_message=str(plp_e),
                    site_config=site_config,
                    run_context=run_context,
                )
                analysis = await self._maybe_analyze_failure(failure_ctx, page=page, run_context=run_context)
                evidence = {"failure_context": failure_ctx}
                if analysis:
                    evidence["failure_analysis"] = analysis
                    # CR-ATELIER-003 Phase D-6: パッチ候補を生成
                    patch = await self._maybe_build_patch_candidate(
                        failure_context=failure_ctx,
                        failure_analysis=analysis,
                        site_config=site_config,
                        run_context=run_context,
                        page=page,
                    )
                    if patch:
                        evidence["self_healing_patch_candidate"] = patch
                return DiscoveryResult(
                    ok=False,
                    site=site,
                    query=query,
                    message=f"PlpDriver failed: {str(plp_e)}",
                    evidence=evidence,
                )

        # Step 6: pdp_links が存在する場合の処理
        # CR-ATELIER-003 Phase C-4 Step 3: pdp_links>0 の場合の通常フローを実装
        if pdp_links:
            self.log.debug(f"[Orchestrator] Found {len(pdp_links)} PDP links, extracting from PDP list...")
            # CR-ATELIER-003 Phase D-4: Telemetry に pdp_links 一覧を記録
            if telemetry and hasattr(telemetry, 'save_json'):
                try:
                    from app.agents.browser.telemetry import TelemetryContext
                    tctx = TelemetryContext(site=site, query=query, run_id=run_context.run_id, stage="plp_pdp_links_found")
                    await telemetry.save_json(
                        "plp_pdp_links",
                        {
                            "entry_url": target_url,
                            "current_url": page.url,
                            "pdp_links_count": len(pdp_links),
                            "pdp_links": pdp_links[:10],  # 最初の10件のみ記録
                        },
                        tctx,
                    )
                except Exception as te:
                    self.log.warning(f"[Orchestrator] Failed to record PDP links: {te}", exc_info=True)
            
            try:
                # BrowserExtractionService をインスタンス化
                extraction_service = BrowserExtractionService(self.log, self.runtime_kwargs)
                
                # prepare_hook を構築（BrowserUseAgent._build_pdp_prepare_hook と同じロジック）
                async def prepare_hook(page: Page):
                    """PDP ページの準備フック（BrowserUseAgent._build_pdp_prepare_hook と同じ）"""
                    # UI helpers をインポート（CR-ATELIER-003 Phase C）
                    try:
                        from app.agents.browser import ui_helpers
                        if ui_helpers.kill_overlays:
                            await ui_helpers.kill_overlays(page)
                        if ui_helpers.click_continue_shopping_if_present:
                            await ui_helpers.click_continue_shopping_if_present(page, site_config)
                        if ui_helpers.dismiss_geo_modal:
                            await ui_helpers.dismiss_geo_modal(page, self.log)
                    except Exception as e:
                        self.log.warning(f"[Orchestrator] UI helpers failed: {e}", exc_info=True)
                    
                    # Visual regression check（設定されている場合）
                    if settings.get("enable_visual_regression_check") and "pdp" in (settings.get("vrt_scope") or ""):
                        try:
                            from app.utils.visual_regression import compare_and_maybe_update
                            await compare_and_maybe_update(page, "pdp", settings)
                        except Exception as e:
                            self.log.warning(f"[Orchestrator] Visual regression check failed: {e}", exc_info=True)
                
                # extract_from_pdp_list を呼び出す（BrowserUseAgent._run_plp_flow と同じ）
                result = await extraction_service.extract_from_pdp_list(
                    page=page,
                    context=context,
                    site=site,
                    query=query,
                    pdp_links=pdp_links,
                    site_config=site_config,
                    settings=settings,
                    run_context=run_context,
                    start_t=start_t,
                    budget_ms=budget_ms,
                    prepare_page=prepare_hook,
                )
                
                # DiscoveryResult を返す
                return result
                
            except Exception as extract_e:
                self.log.error(f"[Orchestrator] extract_from_pdp_list failed: {extract_e}", exc_info=True)
                # エラーが発生した場合、DiscoveryResult を返す
                # CR-ATELIER-003 Phase D-4: failure_context を標準化
                failure_ctx = self._build_failure_context(
                    site=site,
                    query=query,
                    final_url=page.url,
                    error_type="pdp_extraction_failed",
                    error_class=type(extract_e).__name__,
                    error_message=str(extract_e),
                    site_config=site_config,
                    run_context=run_context,
                )
                analysis = await self._maybe_analyze_failure(failure_ctx, page=page, run_context=run_context)
                evidence = {"failure_context": failure_ctx}
                if analysis:
                    evidence["failure_analysis"] = analysis
                    # CR-ATELIER-003 Phase D-6: パッチ候補を生成
                    patch = await self._maybe_build_patch_candidate(
                        failure_context=failure_ctx,
                        failure_analysis=analysis,
                        site_config=site_config,
                        run_context=run_context,
                        page=page,
                    )
                    if patch:
                        evidence["self_healing_patch_candidate"] = patch
                return DiscoveryResult(
                    ok=False,
                    site=site,
                    query=query,
                    message=f"extract_from_pdp_list failed: {str(extract_e)}",
                    evidence=evidence,
                )
        
        # pdp_links が空で、PlpDriver も呼ばれなかった場合（通常は発生しない）
        self.log.error("[Orchestrator] No PDP links found and PlpDriver fallback was not called")
        # CR-ATELIER-003 Phase D-4: failure_context を標準化
        failure_ctx = self._build_failure_context(
            site=site,
            query=query,
            final_url=page.url,
            error_type="no_pdp_links",
            error_class="NoPdpLinksError",
            error_message="No PDP links found and all fallback strategies failed",
            site_config=site_config,
            run_context=run_context,
        )
        analysis = await self._maybe_analyze_failure(failure_ctx, page=page, run_context=run_context)
        evidence = {"failure_context": failure_ctx}
        if analysis:
            evidence["failure_analysis"] = analysis
            # CR-ATELIER-003 Phase D-6: パッチ候補を生成
            patch = await self._maybe_build_patch_candidate(
                failure_context=failure_ctx,
                failure_analysis=analysis,
                site_config=site_config,
                run_context=run_context,
                page=page,
            )
            if patch:
                evidence["self_healing_patch_candidate"] = patch
        return DiscoveryResult(
            ok=False,
            site=site,
            query=query,
            message="No PDP links found and all fallback strategies failed",
            evidence=evidence,
        )

    async def run_pdp(
        self,
        *,
        page: Page,
        context: BrowserContext,
        site: str,
        query: str,
        site_config: Dict[str, Any],
        settings: Dict[str, Any],
        run_context: RunContext,
        target_url: Optional[str] = None,
    ) -> DiscoveryResult:
        """
        PDP 抽出フローを統括する。
        - 単一PDP抽出: extract_single_pdp を呼び出す
        
        CR-ATELIER-003 Phase D-2: BrowserUseAgent._run_pdp_flow から移行

        Args:
            page: Playwright Page オブジェクト
            context: Playwright BrowserContext オブジェクト
            site: サイトコード
            query: 検索クエリ
            site_config: サイト設定
            settings: 実行設定
            run_context: RunContext オブジェクト
            target_url: PDP URL（None の場合は page.url を使用）

        Returns:
            DiscoveryResult
        """
        # BrowserExtractionService をインスタンス化
        extraction_service = BrowserExtractionService(self.log, self.runtime_kwargs)
        
        # prepare_page フックを構築（BrowserUseAgent._build_pdp_prepare_hook と同等の処理）
        # CR-ATELIER-003 Phase D-4: Telemetry を取得して prepare_hook に渡す
        telemetry_client = None
        if TelemetryClient is not None:
            try:
                telemetry_client = TelemetryClient(run_context=run_context)
            except Exception as te:
                self.log.warning(f"[Orchestrator] Failed to create TelemetryClient: {te}", exc_info=True)
        
        async def prepare_hook(inner_page: Page) -> None:
            """PDP ページの準備フック（BrowserUseAgent._build_pdp_prepare_hook と同等）"""
            # CR-ATELIER-003 Phase D-4: PDP DOM を Telemetry に保存
            if telemetry_client and hasattr(telemetry_client, 'record_plp_state'):
                try:
                    await telemetry_client.record_plp_state(
                        inner_page,
                        name="pdp_dom",
                        site_config=site_config,
                    )
                except Exception as te:
                    self.log.warning(f"[Orchestrator] Failed to record PDP DOM: {te}", exc_info=True)
            
            # UI helpers をインポート（CR-ATELIER-003 Phase C）
            try:
                from app.agents.browser import ui_helpers
                if ui_helpers.kill_overlays:
                    await ui_helpers.kill_overlays(inner_page)
                if ui_helpers.click_continue_shopping_if_present:
                    await ui_helpers.click_continue_shopping_if_present(inner_page, site_config)
                if ui_helpers.dismiss_geo_modal:
                    await ui_helpers.dismiss_geo_modal(inner_page, self.log)
            except Exception as e:
                self.log.warning(f"[Orchestrator] UI helpers failed: {e}", exc_info=True)
            
            # Visual regression check（設定されている場合）
            if settings.get("enable_visual_regression_check") and "pdp" in (settings.get("vrt_scope") or ""):
                try:
                    from app.utils.visual_regression import compare_and_maybe_update
                    vrt_result = await compare_and_maybe_update(inner_page, "pdp", settings)
                    # CR-ATELIER-003 Phase D-4: VRT 結果を Telemetry に記録（可能であれば）
                    if telemetry_client and hasattr(telemetry_client, 'save_json') and vrt_result:
                        try:
                            from app.agents.browser.telemetry import TelemetryContext
                            tctx = TelemetryContext(site=site, query=query, run_id=run_context.run_id, stage="pdp_vrt")
                            await telemetry_client.save_json(
                                "pdp_vrt_result",
                                {
                                    "url": inner_page.url,
                                    "vrt_result": str(vrt_result),
                                },
                                tctx,
                            )
                        except Exception as te:
                            self.log.warning(f"[Orchestrator] Failed to record VRT result: {te}", exc_info=True)
                except Exception as e:
                    self.log.warning(f"[Orchestrator] Visual regression check failed: {e}", exc_info=True)
        
        # 有効な URL を決定
        effective_url = target_url or page.url
        
        # extract_single_pdp を呼び出す
        # CR-ATELIER-003 Phase D-3: 例外処理は Orchestrator 内で行う
        # - extract_single_pdp は通常 DiscoveryResult を返すが、
        #   価格が見つからない場合などに ValueError を投げる可能性がある
        # - その場合は DiscoveryResult(ok=False, ...) に変換して返す
        try:
            result = await extraction_service.extract_single_pdp(
                page=page,
                context=context,
                site=site,
                query=query,
                settings=settings,
                run_context=run_context,
                site_config=site_config,
                target_url=effective_url,
                prepare_page=prepare_hook,
            )
            return result
        except ValueError as e:
            # 価格が見つからない場合など、extract_single_pdp が ValueError を投げた場合
            self.log.warning(f"[Orchestrator] extract_single_pdp failed: {e}")
            # CR-ATELIER-003 Phase D-4: failure_context を標準化
            failure_ctx = self._build_failure_context(
                site=site,
                query=query,
                final_url=effective_url,
                error_type="pdp_extraction_failed",
                error_class=type(e).__name__,
                error_message=str(e),
                site_config=site_config,
                run_context=run_context,
            )
            analysis = await self._maybe_analyze_failure(failure_ctx, page=page, run_context=run_context)
            evidence = {"final_url": effective_url, "failure_context": failure_ctx}
            if analysis:
                evidence["failure_analysis"] = analysis
                # CR-ATELIER-003 Phase D-6: パッチ候補を生成
                patch = await self._maybe_build_patch_candidate(
                    failure_context=failure_ctx,
                    failure_analysis=analysis,
                    site_config=site_config,
                    run_context=run_context,
                )
                if patch:
                    evidence["self_healing_patch_candidate"] = patch
            return DiscoveryResult(
                ok=False,
                site=site,
                query=query,
                message=str(e),
                evidence=evidence,
            )
        except Exception as e:
            # 予期しない例外が発生した場合
            self.log.error(f"[Orchestrator] Unexpected error in extract_single_pdp: {e}", exc_info=True)
            # CR-ATELIER-003 Phase D-4: failure_context を標準化
            failure_ctx = self._build_failure_context(
                site=site,
                query=query,
                final_url=effective_url,
                error_type="pdp_extraction_failed",
                error_class=type(e).__name__,
                error_message=str(e),
                site_config=site_config,
                run_context=run_context,
            )
            analysis = await self._maybe_analyze_failure(failure_ctx, page=page, run_context=run_context)
            evidence = {"final_url": effective_url, "error": str(e), "failure_context": failure_ctx}
            if analysis:
                evidence["failure_analysis"] = analysis
                # CR-ATELIER-003 Phase D-6: パッチ候補を生成
                patch = await self._maybe_build_patch_candidate(
                    failure_context=failure_ctx,
                    failure_analysis=analysis,
                    site_config=site_config,
                    run_context=run_context,
                )
                if patch:
                    evidence["self_healing_patch_candidate"] = patch
            return DiscoveryResult(
                ok=False,
                site=site,
                query=query,
                message=f"Unexpected error during PDP extraction: {str(e)}",
                evidence=evidence,
            )
    
    def _build_failure_context(
        self,
        *,
        site: str,
        query: str,
        final_url: str,
        error_type: str,
        error_class: str,
        error_message: str,
        site_config: Dict[str, Any],
        run_context: RunContext,
    ) -> Dict[str, Any]:
        """
        CR-ATELIER-003 Phase D-4: failure_context を標準化して構築する
        
        Args:
            site: サイトコード
            query: 検索クエリ
            final_url: 最終URL
            error_type: エラータイプ（"no_pdp_links" / "pdp_extraction_failed" / "trap_recovery_failed" など）
            error_class: 例外クラス名
            error_message: エラーメッセージ
            site_config: サイト設定
            run_context: RunContext オブジェクト
        
        Returns:
            標準化された failure_context 辞書
        """
        failure_ctx = {
            "final_url": final_url,
            "error_type": error_type,
            "error_class": error_class,
            "error_message": error_message,
            "site": site,
            "query": query,
            "run_id": getattr(run_context, "run_id", "unknown"),
        }
        
        # dom_snapshot_path を取得（可能であれば）
        try:
            if hasattr(run_context, "get_path"):
                dom_path = run_context.get_path("failure_dom.html")
                if dom_path:
                    failure_ctx["dom_snapshot_path"] = str(dom_path)
        except Exception:
            pass
        
        # screenshot_paths を取得（可能であれば）
        try:
            if hasattr(run_context, "screenshots") and isinstance(run_context.screenshots, list):
                failure_ctx["screenshot_paths"] = run_context.screenshots[:6]  # 最大6件
            elif hasattr(run_context, "get_path"):
                run_dir = Path(getattr(run_context, "run_path", "."))
                if run_dir.exists():
                    recent_pngs = sorted(run_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
                    failure_ctx["screenshot_paths"] = [str(p) for p in recent_pngs[:6]]
        except Exception:
            pass
        
        # site_config の要約（必要最低限）
        try:
            site_summary = {
                "site_code": site_config.get("site_code") or site_config.get("site") or site,
            }
            # selectors の要約（存在する場合のみ）
            selectors = site_config.get("selectors", {})
            if selectors:
                site_summary["has_plp_selectors"] = bool(selectors.get("plp"))
                site_summary["has_pdp_selectors"] = bool(selectors.get("pdp"))
            failure_ctx["site_config_summary"] = site_summary
        except Exception:
            pass
        
        return failure_ctx
    
    async def _maybe_analyze_failure(
        self,
        failure_context: Dict[str, Any],
        *,
        page: Optional[Page] = None,
        run_context: Optional[RunContext] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        CR-ATELIER-003 Phase D-5: Self-Healing 連携導線（診断まで）
        
        FailureAnalysisAgent を呼び出して失敗を分析し、
        構造化された診断結果を返す。
        
        Args:
            failure_context: 標準化された failure_context 辞書
            page: Playwright Page オブジェクト（オプション）
            run_context: RunContext オブジェクト（オプション）
        
        Returns:
            分析結果の辞書。以下のフィールドを含む:
                - summary: 人間向け要約
                - root_causes: 想定される原因リスト
                - suggested_fixes: 修正案のリスト
                - confidence: 信頼度（0.0-1.0）
            分析が失敗した場合や analysis_agent が None の場合は None を返す。
        """
        self.log.info(
            f"[Orchestrator][SelfHealing] Failure detected: "
            f"type={failure_context.get('error_type')}, "
            f"class={failure_context.get('error_class')}, "
            f"url={failure_context.get('final_url')}"
        )
        
        # analysis_agent が None なら何もせず None を返す
        if self.analysis_agent is None:
            self.log.debug("[Orchestrator][SelfHealing] AnalysisAgent is not available, skipping analysis")
            return None
        
        try:
            # FailureAnalysisAgent の analyze_failure_context メソッドを呼び出す
            if hasattr(self.analysis_agent, 'analyze_failure_context'):
                analysis = await self.analysis_agent.analyze_failure_context(
                    failure_context,
                    run_context=run_context,
                )
                self.log.info(
                    f"[Orchestrator][SelfHealing] Analysis completed: "
                    f"summary={analysis.get('summary', 'N/A')[:100]}, "
                    f"confidence={analysis.get('confidence', 0.0)}"
                )
                return analysis
            else:
                # 既存の analyze メソッドが存在する場合（後方互換性）
                self.log.warning(
                    "[Orchestrator][SelfHealing] AnalysisAgent does not have analyze_failure_context method, "
                    "falling back to legacy analyze method"
                )
                return None
        except Exception as e:
            # 例外発生時はログに出力して None を返す（メインフローには影響させない）
            self.log.error(
                f"[Orchestrator][SelfHealing] Failure analysis failed: {e}",
                exc_info=True
            )
            return None
    
    async def _maybe_build_patch_candidate(
        self,
        *,
        failure_context: Dict[str, Any],
        failure_analysis: Dict[str, Any],
        site_config: Dict[str, Any],
        run_context: RunContext,
        page: Optional[Page] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        CR-ATELIER-003 Phase D-6, D-10: Self-Healing パッチ候補生成
        
        SelfHealingPatchAgent を呼び出してパッチ候補を生成・保存する。
        Phase D-10: SelectorRepairAgent による selector repair も実行する。
        
        Args:
            failure_context: 標準化された failure_context 辞書
            failure_analysis: FailureAnalysisAgent による分析結果
            site_config: サイト設定
            run_context: RunContext オブジェクト
            page: Playwright Page オブジェクト（オプション、DOM snapshot 取得用）
        
        Returns:
            生成したパッチ候補の辞書。生成に失敗した場合や patch_agent が None の場合は None。
        """
        # patch_agent が None なら何もせず None を返す
        if self.patch_agent is None:
            self.log.debug("[Orchestrator][SelfHealing] PatchAgent is not available, skipping patch generation")
            return None
        
        try:
            # CR-ATELIER-003 Phase D-10: SelectorRepairAgent による selector repair
            selector_repair_result = None
            if self.selector_repair_agent and self.policy:
                # Policy で selector auto-healing が有効かチェック
                if self.policy.selector_auto_healing_enabled():
                    selector_repair_result = await self._maybe_repair_selectors(
                        failure_context=failure_context,
                        failure_analysis=failure_analysis,
                        site_config=site_config,
                        run_context=run_context,
                        page=page,
                    )
            
            # SelfHealingPatchAgent.build_patch_candidate を呼び出す
            patch_candidate = await self.patch_agent.build_patch_candidate(
                failure_context=failure_context,
                failure_analysis=failure_analysis,
                site_config=site_config,
                run_context=run_context,
                selector_repair_result=selector_repair_result,
            )
            
            if patch_candidate:
                self.log.info(
                    f"[Orchestrator][SelfHealing] Patch candidate generated: "
                    f"run_id={patch_candidate.get('run_id')}, "
                    f"changes_count={len(patch_candidate.get('changes', []))}"
                )
            
            return patch_candidate
            
        except Exception as e:
            # 例外発生時はログに出力して None を返す（メインフローには影響させない）
            self.log.error(
                f"[Orchestrator][SelfHealing] Patch candidate generation failed: {e}",
                exc_info=True
            )
            return None
    
    async def _maybe_repair_selectors(
        self,
        *,
        failure_context: Dict[str, Any],
        failure_analysis: Dict[str, Any],
        site_config: Dict[str, Any],
        run_context: RunContext,
        page: Optional[Page] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        CR-ATELIER-003 Phase D-10: SelectorRepairAgent による selector repair
        
        Args:
            failure_context: 標準化された failure_context 辞書
            failure_analysis: FailureAnalysisAgent による分析結果
            site_config: サイト設定
            run_context: RunContext オブジェクト
            page: Playwright Page オブジェクト（オプション）
        
        Returns:
            SelectorRepairAgent.propose_selector_patches() の戻り値。失敗時は None。
        """
        if self.selector_repair_agent is None:
            return None
        
        try:
            site = failure_context.get("site", "unknown")
            error_type = failure_context.get("error_type", "")
            error_message = failure_context.get("error_message", "")
            
            # selector 関連エラーかどうかを判定
            is_selector_error = (
                "selector" in error_message.lower() or
                "not found" in error_message.lower() or
                error_type in ("pdp_extraction_failed", "plp_extraction_failed")
            )
            
            if not is_selector_error:
                self.log.debug(
                    f"[Orchestrator][SelectorRepair] Skipping selector repair: "
                    f"error_type={error_type} is not selector-related"
                )
                return None
            
            # page_type を判定（error_type から推測）
            page_type = "pdp" if "pdp" in error_type.lower() else "plp"
            
            # DOM snapshot を取得
            dom_snapshot_html = ""
            dom_snapshot_path = failure_context.get("dom_snapshot_path")
            if dom_snapshot_path:
                try:
                    dom_path = Path(dom_snapshot_path)
                    if dom_path.exists():
                        dom_snapshot_html = dom_path.read_text(encoding="utf-8")
                except Exception as e:
                    self.log.warning(f"[Orchestrator][SelectorRepair] Failed to read DOM snapshot: {e}")
            
            # page から直接取得を試みる
            if not dom_snapshot_html and page and not page.is_closed():
                try:
                    dom_snapshot_html = await page.content()
                except Exception as e:
                    self.log.warning(f"[Orchestrator][SelectorRepair] Failed to get page content: {e}")
            
            if not dom_snapshot_html:
                self.log.warning("[Orchestrator][SelectorRepair] DOM snapshot not available, skipping selector repair")
                return None
            
            # 現行の selectors を取得
            current_selectors = site_config.get("selectors", {}).get(page_type, {})
            
            # CR-ATELIER-003 Phase D-10.1 Integration: Sandbox から過去の成功/失敗履歴を取得
            previous_successes = []
            previous_failures = []
            if self.sandbox:
                try:
                    previous_successes = self.sandbox.get_previous_successes(
                        site=site,
                        kind="pdp_link",
                    )
                    previous_failures = self.sandbox.get_previous_failures(
                        site=site,
                        kind="pdp_link",
                    )
                    self.log.debug(
                        f"[Orchestrator][SelectorFeedback] Loaded feedback: "
                        f"successes={len(previous_successes)}, failures={len(previous_failures)}"
                    )
                except Exception as e:
                    self.log.warning(
                        f"[Orchestrator][SelectorFeedback] Failed to load feedback: {e}",
                        exc_info=True
                    )
            
            # SelectorRepairAgent を呼び出し
            selector_repair_result = await self.selector_repair_agent.propose_selector_patches(
                site=site,
                page_type=page_type,
                failure_context=failure_context,
                failure_analysis=failure_analysis,
                dom_snapshot_html=dom_snapshot_html,
                current_selectors=current_selectors,
                site_config=site_config,  # CR-ATELIER-003 Phase D-10.1: サイト固有制約抽出用
                previous_successes=previous_successes,  # CR-ATELIER-003 Phase D-10.1 Integration
                previous_failures=previous_failures,  # CR-ATELIER-003 Phase D-10.1 Integration
            )
            
            if selector_repair_result and selector_repair_result.get("candidates"):
                self.log.info(
                    f"[Orchestrator][SelectorRepair] Generated {len(selector_repair_result['candidates'])} "
                    f"selector candidates for {site}/{page_type}"
                )
            
            return selector_repair_result
            
        except Exception as e:
            self.log.error(
                f"[Orchestrator][SelectorRepair] Selector repair failed: {e}",
                exc_info=True
            )
            return None


    async def run_with_self_healing_once(
        self,
        *,
        page: Page,
        context: BrowserContext,
        site: str,
        query: str,
        site_config: Dict[str, Any],
        settings: Dict[str, Any],
        run_context: RunContext,
        target_url: str,
        start_t: float,
        budget_ms: int,
        overrides_json: Dict[str, Any],
        nav_outcome: Optional[NavigationOutcome] = None,
        trap_checker: Optional[Callable[[str], bool]] = None,
        telemetry: Optional[Any] = None,
        plugin: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Self-Healing Loop v1: 初回失敗 → patch 適用 → 再実行
        
        CR-ATELIER-003 Phase D-8: Self-Healing 自動適用フロー（第一段階）
        
        Args:
            page: Playwright Page オブジェクト
            context: Playwright BrowserContext オブジェクト
            site: サイトコード
            query: 検索クエリ
            site_config: サイト設定（本番）
            settings: 実行設定
            run_context: RunContext オブジェクト
            target_url: ターゲットURL
            start_t: 開始時刻
            budget_ms: 予算時間（ミリ秒）
            overrides_json: 現行の overrides.local.json の内容（辞書形式）
            nav_outcome: NavigationDriver の実行結果（オプション）
            trap_checker: Trap 判定関数（オプション）
            telemetry: TelemetryClient（オプション）
            plugin: StrategyPlugin（オプション）
        
        Returns:
            辞書。以下のフィールドを含む:
                - initial: DiscoveryResult（初回実行結果）
                - after_patch: Optional[DiscoveryResult]（パッチ適用後の再実行結果）
                - self_healing_success: bool（Self-Healing が成功したか）
                - patch_candidate: Optional[Dict]（生成されたパッチ候補）
                - sandbox_config: Optional[Dict]（Sandbox で適用された site_config）
        """
        # Step 1: 通常の run_plp_to_pdp → run_pdp を実行
        self.log.info("[Orchestrator][SelfHealing] Starting initial run...")
        
        initial_result_plp = await self.run_plp_to_pdp(
            page=page,
            context=context,
            site=site,
            query=query,
            site_config=site_config,
            settings=settings,
            run_context=run_context,
            target_url=target_url,
            start_t=start_t,
            budget_ms=budget_ms,
            nav_outcome=nav_outcome,
            trap_checker=trap_checker,
            telemetry=telemetry,
            plugin=plugin,
        )
        
        # PlpNavigationResult の場合は DiscoveryResult に変換
        if isinstance(initial_result_plp, PlpNavigationResult):
            # PLP 成功 → PDP 抽出を実行
            if initial_result_plp.pdp_url:
                initial_result = await self.run_pdp(
                    page=page,
                    context=context,
                    site=site,
                    query=query,
                    site_config=site_config,
                    settings=settings,
                    run_context=run_context,
                    target_url=initial_result_plp.pdp_url,
                )
            else:
                # pdp_url が無い場合は失敗として扱う
                initial_result = DiscoveryResult(
                    ok=False,
                    site=site,
                    query=query,
                    evidence={
                        "error": "No PDP URL found",
                        "plp_result": initial_result_plp,
                    },
                )
        else:
            # すでに DiscoveryResult の場合
            initial_result = initial_result_plp
        
        # Step 2: 初回実行が成功した場合、Self-Healing 不要
        if initial_result.ok:
            self.log.info("[Orchestrator][SelfHealing] Initial run succeeded. Self-Healing not needed.")
            return {
                "initial": initial_result,
                "after_patch": None,
                "self_healing_success": False,
                "patch_candidate": None,
                "sandbox_config": None,
            }
        
        # Step 3: 失敗した場合、patch_candidate を生成（既存機構）
        self.log.info("[Orchestrator][SelfHealing] Initial run failed. Generating patch candidate...")
        
        # failure_context を構築
        failure_context = self._build_failure_context(
            site=site,
            query=query,
            error_type="pdp_extraction_failed",
            error_class=type(initial_result.evidence.get("error", Exception())).__name__,
            error_message=str(initial_result.evidence.get("error", "Unknown error")),
            final_url=page.url,
            run_context=run_context,
            site_config=site_config,
        )
        
        # failure_analysis を取得（既存の evidence から）
        failure_analysis = initial_result.evidence.get("failure_analysis")
        if not failure_analysis:
            # failure_analysis が無い場合は空の辞書を使用
            failure_analysis = {
                "summary": "Initial run failed",
                "root_causes": [],
                "suggested_fixes": [],
            }
        
        # patch_candidate を生成
        patch_candidate = None
        if self.patch_agent:
            patch_candidate = await self._maybe_build_patch_candidate(
                failure_context=failure_context,
                failure_analysis=failure_analysis,
                site_config=site_config,
                run_context=run_context,
                page=None,  # run_with_self_healing_loop では page が利用できない場合がある
            )
        
        if not patch_candidate:
            self.log.warning("[Orchestrator][SelfHealing] Patch candidate generation failed or skipped.")
            return {
                "initial": initial_result,
                "after_patch": None,
                "self_healing_success": False,
                "patch_candidate": None,
                "sandbox_config": None,
            }
        
        # Step 4: SelfHealingSandbox に適用
        if not self.sandbox:
            self.log.warning("[Orchestrator][SelfHealing] Sandbox not available. Skipping Self-Healing.")
            return {
                "initial": initial_result,
                "after_patch": None,
                "self_healing_success": False,
                "patch_candidate": patch_candidate,
                "sandbox_config": None,
            }
        
        self.log.info("[Orchestrator][SelfHealing] Applying patch in sandbox...")
        sandbox_overrides = self.sandbox.apply_patch_in_memory(
            overrides_json=overrides_json,
            patch_candidate=patch_candidate,
        )
        
        # Sandbox から site_config を取得
        sandbox_config = self.sandbox.get_site_config_from_overrides(
            overrides_json=sandbox_overrides,
            site_code=site,
        )
        
        if not sandbox_config:
            self.log.warning(f"[Orchestrator][SelfHealing] Site '{site}' not found in sandbox overrides.")
            return {
                "initial": initial_result,
                "after_patch": None,
                "self_healing_success": False,
                "patch_candidate": patch_candidate,
                "sandbox_config": None,
            }
        
        # Step 5: 仮想 site_config で再度 run_plp_to_pdp → run_pdp を実行
        self.log.info("[Orchestrator][SelfHealing] Re-running with sandbox config...")
        
        after_patch_result_plp = await self.run_plp_to_pdp(
            page=page,
            context=context,
            site=site,
            query=query,
            site_config=sandbox_config,
            settings=settings,
            run_context=run_context,
            target_url=target_url,
            start_t=start_t,
            budget_ms=budget_ms,
            nav_outcome=None,  # 再実行なので nav_outcome は None
            trap_checker=trap_checker,
            telemetry=telemetry,
            plugin=plugin,
        )
        
        # PlpNavigationResult の場合は DiscoveryResult に変換
        if isinstance(after_patch_result_plp, PlpNavigationResult):
            # PLP 成功 → PDP 抽出を実行
            if after_patch_result_plp.pdp_url:
                after_patch_result = await self.run_pdp(
                    page=page,
                    context=context,
                    site=site,
                    query=query,
                    site_config=sandbox_config,
                    settings=settings,
                    run_context=run_context,
                    target_url=after_patch_result_plp.pdp_url,
                )
            else:
                # pdp_url が無い場合は失敗として扱う
                after_patch_result = DiscoveryResult(
                    ok=False,
                    site=site,
                    query=query,
                    evidence={
                        "error": "No PDP URL found after patch",
                        "plp_result": after_patch_result_plp,
                    },
                )
        else:
            # すでに DiscoveryResult の場合
            after_patch_result = after_patch_result_plp
        
        self_healing_success = after_patch_result.ok
        
        # CR-ATELIER-003 Phase D-10.1 Integration: selector 使用状況を記録
        if self.sandbox and patch_candidate:
            self._record_selector_usage(
                site=site,
                patch_candidate=patch_candidate,
                success=self_healing_success,
                result=after_patch_result,
            )
        
        if self_healing_success:
            self.log.info("[Orchestrator][SelfHealing] Self-Healing succeeded!")
        else:
            self.log.warning("[Orchestrator][SelfHealing] Self-Healing failed. Manual intervention may be needed.")
        
        return {
            "initial": initial_result,
            "after_patch": after_patch_result,
            "self_healing_success": self_healing_success,
            "patch_candidate": patch_candidate,
            "sandbox_config": sandbox_config,
        }
    
    def _record_selector_usage(
        self,
        *,
        site: str,
        patch_candidate: Dict[str, Any],
        success: bool,
        result: Any,
    ) -> None:
        """
        CR-ATELIER-003 Phase D-10.1 Integration: selector 使用状況を Sandbox に記録する
        
        Args:
            site: サイトコード
            patch_candidate: パッチ候補（selector_patch を含む可能性がある）
            success: 再実行が成功したかどうか
            result: 再実行結果（DiscoveryResult）
        """
        if not self.sandbox:
            return
        
        # patch_candidate から selector_patch を抽出
        changes = patch_candidate.get("changes", [])
        selector_patches = [
            change for change in changes
            if change.get("action") == "selector_patch"
        ]
        
        if not selector_patches:
            # selector_patch がない場合は記録しない
            return
        
        # 各 selector_patch を記録
        for change in selector_patches:
            new_selector = change.get("new_value")
            if not new_selector:
                continue
            
            # セレクタがリスト形式の場合、最初の要素を使用
            if isinstance(new_selector, list):
                if not new_selector:
                    continue
                selector = new_selector[0]
            else:
                selector = str(new_selector)
            
            # Sandbox に記録
            reason = "PDP extraction succeeded" if success else "PDP extraction failed"
            if not success and result:
                # 失敗理由を詳細化
                error_msg = result.evidence.get("error", "")
                if error_msg:
                    reason = f"Failed: {error_msg[:100]}"  # 最大100文字
            
            self.sandbox.record_selector_result(
                site=site,
                selector=selector,
                success=success,
                kind="pdp_link",
                reason=reason,
            )
            
            self.log.debug(
                f"[Orchestrator][SelectorFeedback] Recorded selector result: "
                f"site={site}, selector={selector[:50]}, success={success}"
            )
    
    def _load_overrides_json(self) -> Dict[str, Any]:
        """
        overrides.local.json を読み込む
        
        Returns:
            Dict[str, Any]: overrides.local.json の内容
        """
        
        if not self._overrides_path.exists():
            self.log.warning(
                f"[Orchestrator] overrides.local.json not found at {self._overrides_path}. "
                "Returning empty dict."
            )
            return {}
        
        try:
            with open(self._overrides_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.log.error(
                f"[Orchestrator] Failed to load overrides.local.json: {e}",
                exc_info=True
            )
            return {}
    
    def _load_site_config_from_overrides(self, site: str) -> Optional[Dict[str, Any]]:
        """
        overrides.local.json から特定サイトの site_config を取得
        
        Args:
            site: サイトコード
        
        Returns:
            Optional[Dict[str, Any]]: site_config（存在しない場合は None）
        """
        overrides_json = self._load_overrides_json()
        
        if not overrides_json:
            return None
        
        # overrides.local.json の構造: {"sites": {"MONCLER_OFFICIAL": {...}}}
        sites = overrides_json.get("sites", {})
        site_config = sites.get(site)
        
        if not site_config:
            self.log.warning(
                f"[Orchestrator] Site '{site}' not found in overrides.local.json"
            )
            return None
        
        return site_config
    
    async def _run_full(
        self,
        *,
        page: Page,
        context: BrowserContext,
        site: str,
        query: str,
        site_config: Dict[str, Any],
        settings: Dict[str, Any],
        run_context: RunContext,
        target_url: str,
        start_t: float,
        budget_ms: int,
        nav_outcome: Optional[NavigationOutcome] = None,
        trap_checker: Optional[Callable[[str], bool]] = None,
        telemetry: Optional[Any] = None,
        plugin: Optional[Any] = None,
    ) -> DiscoveryResult:
        """
        run_plp_to_pdp + run_pdp を実行する共通ヘルパ
        
        Args:
            page: Playwright Page オブジェクト
            context: Playwright BrowserContext オブジェクト
            site: サイトコード
            query: 検索クエリ
            site_config: サイト設定
            settings: 実行設定
            run_context: RunContext オブジェクト
            target_url: ターゲットURL
            start_t: 開始時刻
            budget_ms: 予算時間（ミリ秒）
            nav_outcome: NavigationDriver の実行結果（オプション）
            trap_checker: Trap 判定関数（オプション）
            telemetry: TelemetryClient（オプション）
            plugin: StrategyPlugin（オプション）
        
        Returns:
            DiscoveryResult: 実行結果
        """
        # Step 1: run_plp_to_pdp を実行
        result_plp = await self.run_plp_to_pdp(
            page=page,
            context=context,
            site=site,
            query=query,
            site_config=site_config,
            settings=settings,
            run_context=run_context,
            target_url=target_url,
            start_t=start_t,
            budget_ms=budget_ms,
            nav_outcome=nav_outcome,
            trap_checker=trap_checker,
            telemetry=telemetry,
            plugin=plugin,
        )
        
        # Step 2: PlpNavigationResult の場合は DiscoveryResult に変換
        if isinstance(result_plp, PlpNavigationResult):
            # PLP 成功 → PDP 抽出を実行
            if result_plp.pdp_url:
                result = await self.run_pdp(
                    page=page,
                    context=context,
                    site=site,
                    query=query,
                    site_config=site_config,
                    settings=settings,
                    run_context=run_context,
                    target_url=result_plp.pdp_url,
                )
            else:
                # pdp_url が無い場合は失敗として扱う
                result = DiscoveryResult(
                    ok=False,
                    site=site,
                    query=query,
                    evidence={
                        "error": "No PDP URL found",
                        "plp_result": result_plp,
                    },
                )
        else:
            # すでに DiscoveryResult の場合
            result = result_plp
        
        # CR-E2E-001: 成功段階を計算して evidence に追加
        result = self._enrich_result_with_success_stage(
            result=result,
            run_context=run_context,
            nav_outcome=nav_outcome,
            page=page,
        )
        
        return result
    
    async def run_with_self_healing_loop(
        self,
        *,
        page: Page,
        context: BrowserContext,
        site: str,
        query: str,
        site_config: Dict[str, Any],
        settings: Dict[str, Any],
        run_context: RunContext,
        target_url: str,
        start_t: float,
        budget_ms: int,
        max_attempts: Optional[int] = None,
        nav_outcome: Optional[NavigationOutcome] = None,
        trap_checker: Optional[Callable[[str], bool]] = None,
        telemetry: Optional[Any] = None,
        plugin: Optional[Any] = None,
    ) -> DiscoveryResult:
        """
        Self-Healing Loop v2: 最大 N 回の自動再試行 + 本番適用
        
        CR-ATELIER-003 Phase D-9: Self-Healing 本番適用・自動再試行ループ
        
        Args:
            page: Playwright Page オブジェクト
            context: Playwright BrowserContext オブジェクト
            site: サイトコード
            query: 検索クエリ
            site_config: サイト設定（本番）
            settings: 実行設定
            run_context: RunContext オブジェクト
            target_url: ターゲットURL
            start_t: 開始時刻
            budget_ms: 予算時間（ミリ秒）
            max_attempts: 最大再試行回数（None の場合は policy.max_attempts() を使用）
            nav_outcome: NavigationDriver の実行結果（オプション）
            trap_checker: Trap 判定関数（オプション）
            telemetry: TelemetryClient（オプション）
            plugin: StrategyPlugin（オプション）
        
        Returns:
            DiscoveryResult: 実行結果（evidence に self_healing_attempts などを含む）
        """
        if not self.policy:
            self.log.warning("[Orchestrator][SelfHealingLoop] Policy not available. Falling back to normal execution.")
            return await self._run_full(
                page=page,
                context=context,
                site=site,
                query=query,
                site_config=site_config,
                settings=settings,
                run_context=run_context,
                target_url=target_url,
                start_t=start_t,
                budget_ms=budget_ms,
                nav_outcome=nav_outcome,
                trap_checker=trap_checker,
                telemetry=telemetry,
                plugin=plugin,
            )
        
        attempt = 0
        max_attempts_value = max_attempts or self.policy.max_attempts()
        current_site_config = site_config
        auto_patches_applied = 0
        patch_backups: List[str] = []
        
        self.log.info(
            f"[Orchestrator][SelfHealingLoop] Starting Self-Healing Loop "
            f"(max_attempts={max_attempts_value})"
        )
        
        while attempt < max_attempts_value:
            attempt += 1
            self.log.info(f"[Orchestrator][SelfHealingLoop] Attempt {attempt}/{max_attempts_value}")
            
            # 通常実行
            result = await self._run_full(
                page=page,
                context=context,
                site=site,
                query=query,
                site_config=current_site_config,
                settings=settings,
                run_context=run_context,
                target_url=target_url,
                start_t=start_t,
                budget_ms=budget_ms,
                nav_outcome=nav_outcome if attempt == 1 else None,
                trap_checker=trap_checker,
                telemetry=telemetry,
                plugin=plugin,
            )
            
            # 成功した場合、attempts を追記して return
            if result.ok:
                self.log.info(
                    f"[Orchestrator][SelfHealingLoop] Success on attempt {attempt}"
                )
                result.evidence["self_healing_attempts"] = attempt
                result.evidence["auto_patches_applied"] = auto_patches_applied
                result.evidence["patch_backups"] = patch_backups
                
                # メトリクス保存
                await self._record_self_healing_metrics(
                    run_id=run_context.run_id,
                    site=site,
                    self_healing_attempts=attempt,
                    auto_patches_applied=auto_patches_applied,
                    patch_backups=patch_backups,
                    final_status="success",
                    run_context=run_context,
                )
                
                return result
            
            # 失敗した場合、patch_candidate を取得
            patch_candidate = result.evidence.get("self_healing_patch_candidate")
            if not patch_candidate:
                self.log.warning(
                    f"[Orchestrator][SelfHealingLoop] No patch_candidate found. "
                    "Stopping Self-Healing loop."
                )
                break
            
            # Policy チェック
            if not self.policy.enabled():
                self.log.warning(
                    "[Orchestrator][SelfHealingLoop] Policy disabled. Stopping Self-Healing loop."
                )
                break
            
            if not self.policy.is_allowed_site(site):
                self.log.warning(
                    f"[Orchestrator][SelfHealingLoop] Site '{site}' not allowed. "
                    "Stopping Self-Healing loop."
                )
                break
            
            if not self.policy.safe_change(patch_candidate):
                self.log.warning(
                    "[Orchestrator][SelfHealingLoop] Unsafe change detected. "
                    "Stopping Self-Healing loop."
                )
                break
            
            if not self.policy.can_apply_today(site):
                self.log.warning(
                    f"[Orchestrator][SelfHealingLoop] Daily limit reached for '{site}'. "
                    "Stopping Self-Healing loop."
                )
                break
            
            # Sandbox 実行
            if not self.sandbox:
                self.log.warning(
                    "[Orchestrator][SelfHealingLoop] Sandbox not available. "
                    "Stopping Self-Healing loop."
                )
                break
            
            overrides_json = self._load_overrides_json()
            if not overrides_json:
                self.log.warning(
                    "[Orchestrator][SelfHealingLoop] Failed to load overrides_json. "
                    "Stopping Self-Healing loop."
                )
                break
            
            self.log.info("[Orchestrator][SelfHealingLoop] Applying patch in sandbox...")
            sandbox_overrides = self.sandbox.apply_patch_in_memory(
                overrides_json=overrides_json,
                patch_candidate=patch_candidate,
            )
            
            sandbox_site_config = self.sandbox.get_site_config_from_overrides(
                overrides_json=sandbox_overrides,
                site_code=site,
            )
            
            if not sandbox_site_config:
                self.log.warning(
                    f"[Orchestrator][SelfHealingLoop] Site '{site}' not found in sandbox overrides. "
                    "Stopping Self-Healing loop."
                )
                break
            
            # Sandbox で再実行
            sandbox_result = await self._run_full(
                page=page,
                context=context,
                site=site,
                query=query,
                site_config=sandbox_site_config,
                settings=settings,
                run_context=run_context,
                target_url=target_url,
                start_t=start_t,
                budget_ms=budget_ms,
                nav_outcome=None,
                trap_checker=trap_checker,
                telemetry=telemetry,
                plugin=plugin,
            )
            
            # Sandbox が失敗した場合、本番適用しない
            if not sandbox_result.ok:
                self.log.warning(
                    "[Orchestrator][SelfHealingLoop] Sandbox execution failed. "
                    "Not applying patch to production."
                )
                break
            
            # 本番適用
            if not self.patch_applier:
                self.log.warning(
                    "[Orchestrator][SelfHealingLoop] PatchApplier not available. "
                    "Stopping Self-Healing loop."
                )
                break
            
            candidate_path = run_context.run_path / "patch_candidate_self_healing.json"
            backup_suffix = f".bak-selfheal-{attempt}"
            
            try:
                apply_meta = self.patch_applier.apply_patch_candidate(
                    candidate_path=candidate_path,
                    overrides_path=self._overrides_path,
                    backup_suffix=backup_suffix,
                )
                
                if apply_meta and apply_meta.get("success"):
                    auto_patches_applied += 1
                    backup_path = apply_meta.get("backup_path")
                    if backup_path:
                        patch_backups.append(str(backup_path))
                    
                    self.log.info(
                        f"[Orchestrator][SelfHealingLoop] Patch applied successfully "
                        f"(attempt {attempt})"
                    )
                    
                    # 日次カウンタを更新
                    self.policy.increment_daily_counter(site)
                    
                    # 新しい config を読み直してループ続行
                    current_site_config = self._load_site_config_from_overrides(site)
                    if not current_site_config:
                        self.log.warning(
                            f"[Orchestrator][SelfHealingLoop] Failed to reload site_config for '{site}'. "
                            "Stopping Self-Healing loop."
                        )
                        break
                else:
                    self.log.warning(
                        "[Orchestrator][SelfHealingLoop] Patch application failed. "
                        "Stopping Self-Healing loop."
                    )
                    break
            except Exception as e:
                self.log.error(
                    f"[Orchestrator][SelfHealingLoop] Exception during patch application: {e}",
                    exc_info=True
                )
                break
        
        # ループ終了時、最後の result に attempts などを追加
        result.evidence["self_healing_attempts"] = attempt
        result.evidence["auto_patches_applied"] = auto_patches_applied
        result.evidence["patch_backups"] = patch_backups
        
        final_status = "success" if result.ok else "failure"
        
        # CR-E2E-001: 成功段階を計算して evidence に追加
        result = self._enrich_result_with_success_stage(
            result=result,
            run_context=run_context,
            nav_outcome=None,  # self-healing loop では nav_outcome は使わない
            page=None,  # self-healing loop では page は使わない
        )
        
        # メトリクス保存
        await self._record_self_healing_metrics(
            run_id=run_context.run_id,
            site=site,
            self_healing_attempts=attempt,
            auto_patches_applied=auto_patches_applied,
            patch_backups=patch_backups,
            final_status=final_status,
            run_context=run_context,
        )
        
        return result
    
    def _enrich_result_with_success_stage(
        self,
        result: DiscoveryResult,
        run_context: RunContext,
        nav_outcome: Optional[NavigationOutcome] = None,
        page: Optional[Page] = None,
    ) -> DiscoveryResult:
        """
        CR-E2E-001: DiscoveryResult に成功段階情報を追加する
        
        Args:
            result: DiscoveryResult
            run_context: RunContext
            nav_outcome: NavigationOutcome（オプション）
            page: Playwright Page（オプション）
        
        Returns:
            DiscoveryResult: 成功段階情報が追加された結果
        """
        if not compute_success_stage or not collect_run_artifacts:
            return result
        
        try:
            # 実行結果情報を収集
            artifacts = collect_run_artifacts(
                result=result,
                run_context=run_context,
                nav_outcome=nav_outcome,
                page=page,
            )
            # 成功段階を計算
            success_stage, criteria = compute_success_stage(
                run_artifacts=artifacts,
                run_context=run_context,
            )
            # evidence に追加（既存の evidence を破壊しない）
            if result.evidence is None:
                result.evidence = {}
            result.evidence["success_stage"] = success_stage
            result.evidence["criteria"] = criteria
            result.evidence["urls"] = {
                "plp_url": artifacts.get("plp_url"),
                "pdp_url_sample": artifacts.get("pdp_url_sample"),
            }
            result.evidence["extracted_fields"] = artifacts.get("extracted_fields", [])
            result.evidence["screenshots"] = artifacts.get("screenshots", [])
            result.evidence["dom_snapshots"] = artifacts.get("dom_snapshots", [])
            
            # CR-E2E-003A: link_collection サマリを追加（pdp_link_validation_report.json から読み込む）
            if run_context and hasattr(run_context, "get_path"):
                try:
                    validation_report_path = run_context.get_path("pdp_link_validation_report.json")
                    if validation_report_path.exists():
                        import json
                        with open(validation_report_path, "r", encoding="utf-8") as f:
                            validation_report = json.load(f)
                        # CR-E2E-003A: evidence.link_collection にサマリを追加
                        result.evidence["link_collection"] = {
                            "total_candidates": validation_report.get("total_candidates", 0),
                            "total_valid": validation_report.get("total_valid", 0),
                            "top_reject_reasons": validation_report.get("top_reject_reasons", {}),
                            "sample_candidates": validation_report.get("sample_rejected", [])[:10],  # 最大10件
                        }
                except Exception as e:
                    self.log.debug(f"[Orchestrator] Failed to load pdp_link_validation_report: {e}")
            
            # CR-E2E-001: 暫定ルールに基づいて result.ok を更新（S3以上でtrue）
            # 既存の result.ok を上書きするが、暫定ルールとして明示的に実装
            from app.utils.e2e_success_stage import should_result_be_ok
            result.ok = should_result_be_ok(success_stage)
            
            self.log.info(
                f"[Orchestrator] Success stage computed: {success_stage}, criteria={criteria}, result.ok={result.ok}"
            )
        except Exception as e:
            self.log.warning(f"[Orchestrator] Failed to compute success stage: {e}", exc_info=True)
        
        return result
    
    async def _record_self_healing_metrics(
        self,
        *,
        run_id: str,
        site: str,
        self_healing_attempts: int,
        auto_patches_applied: int,
        patch_backups: List[str],
        final_status: str,
        run_context: Optional[RunContext] = None,
    ) -> None:
        """
        Self-Healing Loop v2 のメトリクスを保存する
        
        CR-ATELIER-003 Phase D-9: Task 4
        CR-ATELIER-003 Phase D-12: run_type と scenario_name を追加
        
        Args:
            run_id: 実行ID
            site: サイトコード
            self_healing_attempts: Self-Healing の試行回数
            auto_patches_applied: 自動適用されたパッチ数
            patch_backups: パッチバックアップのパスリスト
            final_status: 最終ステータス（"success" or "failure"）
            run_context: RunContext オブジェクト（run_type と scenario_name を取得するため）
        """
        
        metrics = {
            "run_id": run_id,
            "site": site,
            "self_healing_attempts": self_healing_attempts,
            "auto_patches_applied": auto_patches_applied,
            "patch_backups": patch_backups,
            "final_status": final_status,
            "timestamp": datetime.now().isoformat(),
            "policy": {},
        }
        
        # CR-ATELIER-003 Phase D-12: run_type と scenario_name を追加（E2E 実行時）
        if run_context:
            if hasattr(run_context, "run_type") and run_context.run_type:
                metrics["run_type"] = run_context.run_type
            if hasattr(run_context, "scenario_name") and run_context.scenario_name:
                metrics["scenario_name"] = run_context.scenario_name
        
        # Policy 情報を追加
        if self.policy and hasattr(self.policy, "max_attempts") and hasattr(self.policy, "data"):
            try:
                metrics["policy"] = {
                    "max_attempts": self.policy.max_attempts(),
                    "max_auto_applies_per_day": self.policy.data.get("max_auto_applies_per_day", 0),                                                                
                }
            except Exception:
                # Policy がモックの場合など、エラーを無視
                metrics["policy"] = {}
        
        # メトリクスファイルのパス
        metrics_path = Path("docs/reports/self_healing_metrics.jsonl")
        
        try:
            # ディレクトリを作成
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            
            # JSONL 形式で追記
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            
            self.log.info(
                f"[Orchestrator][SelfHealingLoop] Metrics saved to {metrics_path}"
            )
        except Exception as e:
            self.log.error(
                f"[Orchestrator][SelfHealingLoop] Failed to save metrics: {e}",
                exc_info=True
            )
