# -*- coding: utf-8 -*-
"""
CR-ATELIER-003 Phase D-5: FailureAnalysisAgent 統合のテスト

BrowserOrchestrator から FailureAnalysisAgent が呼び出され、
failure_analysis が DiscoveryResult.evidence に含まれることを確認する。
"""

import pytest
pytest.importorskip("playwright")
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from typing import Dict, Any

from app.agents.browser_orchestrator import BrowserOrchestrator
from app.models.result_models import DiscoveryResult
from app.agents.browser.plp_driver import PlpNavigationResult


@pytest.fixture
def mock_page():
    """モック Page オブジェクト"""
    page = AsyncMock()
    page.url = "https://example.com/plp"
    page.is_closed.return_value = False
    page.content = AsyncMock(return_value="<html><body>Test</body></html>")
    return page


@pytest.fixture
def mock_context(mock_page):
    """モック BrowserContext オブジェクト"""
    context = AsyncMock()
    context.new_page = AsyncMock(return_value=mock_page)
    return context


@pytest.fixture
def mock_run_context():
    """モック RunContext オブジェクト"""
    from pathlib import Path
    run_context = MagicMock()
    run_context.run_id = "test_run_123"
    run_context.run_path = "/tmp/test_run"
    # get_path は Path オブジェクトを返す（resolve() が呼ばれるため）
    run_context.get_path = Mock(return_value=Path("/tmp/test_run/failure_dom.html"))
    run_context.screenshots = []
    run_context.screenshots_path = Path("/tmp/test_run/screenshots")
    run_context.save_json = Mock()
    run_context.save_content = AsyncMock()
    return run_context


@pytest.fixture
def site_config():
    """テスト用 site_config"""
    return {
        "site_code": "TEST_SITE",
        "selectors": {
            "plp": {
                "tile_selectors": [".product-tile"],
                "pdp_link_selectors": ["a.product-link"],
            },
            "pdp": {
                "title": ["h1.product-title"],
                "price": [".product-price"],
            },
        },
    }


@pytest.fixture
def settings():
    """テスト用 settings"""
    return {
        "timeout_sec": 60,
        "pdp_parallel_limit": 3,
    }


@pytest.fixture
def mock_analysis_agent():
    """モック FailureAnalysisAgent"""
    agent = AsyncMock()
    agent.analyze_failure_context = AsyncMock(
        return_value={
            "summary": "テスト分析結果: PDP 抽出に失敗しました",
            "root_causes": ["セレクタが一致しませんでした", "タイムアウトが発生しました"],
            "suggested_fixes": ["セレクタを更新してください", "タイムアウトを延長してください"],
            "confidence": 0.8,
        }
    )
    return agent


@pytest.fixture
def orchestrator(mock_run_context, mock_analysis_agent):
    """BrowserOrchestrator インスタンス（FailureAnalysisAgent 付き）"""
    return BrowserOrchestrator(
        runtime_kwargs={},
        analysis_agent=mock_analysis_agent,
        log=MagicMock(),
    )


@pytest.mark.asyncio
async def test_run_plp_to_pdp_calls_analysis_agent_on_trap_recovery_failed(
    orchestrator,
    mock_page,
    mock_context,
    mock_run_context,
    site_config,
    settings,
    mock_analysis_agent,
):
    """trap_recovery_failed の場合、FailureAnalysisAgent が呼ばれることを確認"""
    with patch("app.agents.browser_orchestrator.NavigationDriver") as mock_nav_driver_class:
        
        # NavigationDriver のモック（trap_detected=True, recovered=False）
        mock_nav_driver = AsyncMock()
        mock_nav_outcome = MagicMock()
        mock_nav_outcome.entry_url = "https://example.com/plp"
        mock_nav_outcome.pdp_links = []
        mock_nav_outcome.trap_detected = True
        mock_nav_outcome.recovered = False
        mock_nav_outcome.trap_reason = "Trap page detected"
        mock_nav_driver.run_plp_flow = AsyncMock(return_value=mock_nav_outcome)
        mock_nav_driver_class.return_value = mock_nav_driver
        
        # TelemetryClient のモック
        with patch("app.agents.browser_orchestrator.TelemetryClient") as mock_telemetry_class:
            mock_telemetry = AsyncMock()
            mock_telemetry.record_plp_state = AsyncMock()
            mock_telemetry.save_json = AsyncMock()
            mock_telemetry_class.return_value = mock_telemetry
            
            # run_plp_to_pdp を実行
            result = await orchestrator.run_plp_to_pdp(
                page=mock_page,
                context=mock_context,
                site="TEST_SITE",
                query="test query",
                site_config=site_config,
                settings=settings,
                run_context=mock_run_context,
                target_url="https://example.com/plp",
                start_t=0.0,
                budget_ms=60000,
            )
            
            # FailureAnalysisAgent が呼ばれたことを確認
            assert mock_analysis_agent.analyze_failure_context.called
            
            # failure_analysis が evidence に含まれていることを確認
            assert result.ok is False
            assert "failure_context" in result.evidence
            assert "failure_analysis" in result.evidence
            
            failure_analysis = result.evidence["failure_analysis"]
            assert "summary" in failure_analysis
            assert "root_causes" in failure_analysis
            assert "suggested_fixes" in failure_analysis
            assert "confidence" in failure_analysis


@pytest.mark.asyncio
async def test_run_plp_to_pdp_calls_analysis_agent_on_plp_driver_failed(
    orchestrator,
    mock_page,
    mock_context,
    mock_run_context,
    site_config,
    settings,
    mock_analysis_agent,
):
    """PlpDriver 失敗の場合、FailureAnalysisAgent が呼ばれることを確認"""
    with patch("app.agents.browser_orchestrator.NavigationDriver") as mock_nav_driver_class, \
         patch("app.agents.browser_orchestrator.PlpDriver") as mock_plp_driver_class, \
         patch("app.agents.browser_orchestrator.TelemetryClient") as mock_telemetry_class:
        
        # NavigationDriver のモック（pdp_links が空）
        mock_nav_driver = AsyncMock()
        mock_nav_outcome = MagicMock()
        mock_nav_outcome.entry_url = "https://example.com/plp"
        mock_nav_outcome.pdp_links = []
        mock_nav_outcome.trap_detected = False
        mock_nav_outcome.recovered = False
        mock_nav_driver.run_plp_flow = AsyncMock(return_value=mock_nav_outcome)
        mock_nav_driver_class.return_value = mock_nav_driver
        
        # PlpDriver のモック（例外を投げる）
        mock_plp_driver = AsyncMock()
        mock_plp_driver.navigate_to_pdp = AsyncMock(side_effect=Exception("PlpDriver failed"))
        mock_plp_driver_class.return_value = mock_plp_driver
        
        # TelemetryClient のモック
        mock_telemetry = AsyncMock()
        mock_telemetry.save_json = AsyncMock()
        mock_telemetry_class.return_value = mock_telemetry
        
        # run_plp_to_pdp を実行
        result = await orchestrator.run_plp_to_pdp(
            page=mock_page,
            context=mock_context,
            site="TEST_SITE",
            query="test query",
            site_config=site_config,
            settings=settings,
            run_context=mock_run_context,
            target_url="https://example.com/plp",
            start_t=0.0,
            budget_ms=60000,
        )
        
        # FailureAnalysisAgent が呼ばれたことを確認
        assert mock_analysis_agent.analyze_failure_context.called
        
        # failure_analysis が evidence に含まれていることを確認
        assert result.ok is False
        assert "failure_context" in result.evidence
        assert "failure_analysis" in result.evidence


@pytest.mark.asyncio
async def test_run_pdp_calls_analysis_agent_on_extraction_failed(
    orchestrator,
    mock_page,
    mock_context,
    mock_run_context,
    site_config,
    settings,
    mock_analysis_agent,
):
    """PDP 抽出失敗の場合、FailureAnalysisAgent が呼ばれることを確認"""
    with patch("app.agents.browser_orchestrator.BrowserExtractionService") as mock_extraction_service_class, \
         patch("app.agents.browser_orchestrator.TelemetryClient") as mock_telemetry_class:
        
        # BrowserExtractionService のモック（ValueError を投げる）
        mock_extraction_service = AsyncMock()
        mock_extraction_service.extract_single_pdp = AsyncMock(
            side_effect=ValueError("Price not found on PDP.")
        )
        mock_extraction_service_class.return_value = mock_extraction_service
        
        # TelemetryClient のモック
        mock_telemetry = AsyncMock()
        mock_telemetry.record_plp_state = AsyncMock()
        mock_telemetry_class.return_value = mock_telemetry
        
        # run_pdp を実行
        result = await orchestrator.run_pdp(
            page=mock_page,
            context=mock_context,
            site="TEST_SITE",
            query="test query",
            site_config=site_config,
            settings=settings,
            run_context=mock_run_context,
            target_url="https://example.com/product/1",
        )
        
        # FailureAnalysisAgent が呼ばれたことを確認
        assert mock_analysis_agent.analyze_failure_context.called
        
        # failure_analysis が evidence に含まれていることを確認
        assert result.ok is False
        assert "failure_context" in result.evidence
        assert "failure_analysis" in result.evidence
        
        failure_analysis = result.evidence["failure_analysis"]
        assert "summary" in failure_analysis
        assert "root_causes" in failure_analysis
        assert "suggested_fixes" in failure_analysis
        assert "confidence" in failure_analysis


@pytest.mark.asyncio
async def test_analysis_agent_exception_does_not_break_flow(
    orchestrator,
    mock_page,
    mock_context,
    mock_run_context,
    site_config,
    settings,
):
    """FailureAnalysisAgent が例外を投げた場合でも、メインフローが正常に動作することを確認"""
    # FailureAnalysisAgent が例外を投げるように設定
    mock_analysis_agent = AsyncMock()
    mock_analysis_agent.analyze_failure_context = AsyncMock(
        side_effect=Exception("Analysis failed")
    )
    
    orchestrator.analysis_agent = mock_analysis_agent
    
    with patch("app.agents.browser_orchestrator.BrowserExtractionService") as mock_extraction_service_class, \
         patch("app.agents.browser_orchestrator.TelemetryClient") as mock_telemetry_class:
        
        # BrowserExtractionService のモック（ValueError を投げる）
        mock_extraction_service = AsyncMock()
        mock_extraction_service.extract_single_pdp = AsyncMock(
            side_effect=ValueError("Price not found on PDP.")
        )
        mock_extraction_service_class.return_value = mock_extraction_service
        
        # TelemetryClient のモック
        mock_telemetry = AsyncMock()
        mock_telemetry.record_plp_state = AsyncMock()
        mock_telemetry_class.return_value = mock_telemetry
        
        # run_pdp を実行
        result = await orchestrator.run_pdp(
            page=mock_page,
            context=mock_context,
            site="TEST_SITE",
            query="test query",
            site_config=site_config,
            settings=settings,
            run_context=mock_run_context,
            target_url="https://example.com/product/1",
        )
        
        # FailureAnalysisAgent が呼ばれたことを確認
        assert mock_analysis_agent.analyze_failure_context.called
        
        # failure_analysis が evidence に含まれていないことを確認（例外が発生したため）
        assert result.ok is False
        assert "failure_context" in result.evidence
        # failure_analysis は None または存在しない
        assert "failure_analysis" not in result.evidence or result.evidence.get("failure_analysis") is None


@pytest.mark.asyncio
async def test_no_analysis_agent_does_not_break_flow(
    mock_page,
    mock_context,
    mock_run_context,
    site_config,
    settings,
):
    """analysis_agent が None の場合でも、メインフローが正常に動作することを確認"""
    # analysis_agent が None の Orchestrator を作成
    # 注意: BrowserOrchestrator.__init__ は analysis_agent が None の場合でも
    # FailureAnalysisAgent を自動的に作成する可能性があるため、
    # 明示的に None を設定する必要がある
    orchestrator = BrowserOrchestrator(
        runtime_kwargs={},
        analysis_agent=None,
        log=MagicMock(),
    )
    # __init__ 後に明示的に None を設定（自動生成を防ぐため）
    orchestrator.analysis_agent = None
    
    with patch("app.agents.browser_orchestrator.BrowserExtractionService") as mock_extraction_service_class, \
         patch("app.agents.browser_orchestrator.TelemetryClient") as mock_telemetry_class:
        
        # BrowserExtractionService のモック（ValueError を投げる）
        mock_extraction_service = AsyncMock()
        mock_extraction_service.extract_single_pdp = AsyncMock(
            side_effect=ValueError("Price not found on PDP.")
        )
        mock_extraction_service_class.return_value = mock_extraction_service
        
        # TelemetryClient のモック
        mock_telemetry = AsyncMock()
        mock_telemetry.record_plp_state = AsyncMock()
        mock_telemetry_class.return_value = mock_telemetry
        
        # run_pdp を実行
        result = await orchestrator.run_pdp(
            page=mock_page,
            context=mock_context,
            site="TEST_SITE",
            query="test query",
            site_config=site_config,
            settings=settings,
            run_context=mock_run_context,
            target_url="https://example.com/product/1",
        )
        
        # failure_analysis が evidence に含まれていないことを確認（analysis_agent が None のため）
        assert result.ok is False
        assert "failure_context" in result.evidence
        # failure_analysis は None または存在しない
        assert "failure_analysis" not in result.evidence or result.evidence.get("failure_analysis") is None
