# -*- coding: utf-8 -*-
"""
CR-ATELIER-003 Phase D-6: SelfHealingPatchAgent のテスト

パッチ候補生成のロジックと BrowserOrchestrator との統合をテストする。
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from typing import Dict, Any
from pathlib import Path

from app.agents.self_healing_patch_agent import SelfHealingPatchAgent
from app.agents.browser_orchestrator import BrowserOrchestrator
from app.models.result_models import DiscoveryResult


@pytest.fixture
def mock_run_context():
    """モック RunContext オブジェクト"""
    run_context = MagicMock()
    run_context.run_id = "test_run_123"
    run_context.run_path = "/tmp/test_run"
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
        "discovery_settings": {
            "timeout_sec": 60,
        },
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
def patch_agent():
    """SelfHealingPatchAgent インスタンス"""
    return SelfHealingPatchAgent(runtime_kwargs={})


@pytest.mark.asyncio
async def test_build_patch_candidate_navigation_failed_timeout_increase(
    patch_agent,
    mock_run_context,
    site_config,
):
    """navigation_failed の場合、timeout_sec を +30 秒するパッチ候補が生成されることを確認"""
    failure_context = {
        "site": "TEST_SITE",
        "run_id": "test_run_123",
        "error_type": "navigation_failed",
        "error_class": "TimeoutError",
        "error_message": "Navigation timeout",
    }
    
    failure_analysis = {
        "summary": "Navigation timeout occurred",
        "root_causes": ["Site response was slow"],
        "suggested_fixes": ["Increase timeout"],
        "confidence": 0.8,
    }
    
    patch_candidate = await patch_agent.build_patch_candidate(
        failure_context=failure_context,
        failure_analysis=failure_analysis,
        site_config=site_config,
        run_context=mock_run_context,
    )
    
    assert patch_candidate is not None
    assert patch_candidate["target_site"] == "TEST_SITE"
    assert patch_candidate["run_id"] == "test_run_123"
    assert patch_candidate["strategy"] == "heuristic_v1"
    assert len(patch_candidate["changes"]) > 0
    
    # timeout_sec の increase 提案が含まれていることを確認
    timeout_change = next(
        (c for c in patch_candidate["changes"] if c.get("path") == "discovery_settings.timeout_sec"),
        None
    )
    assert timeout_change is not None
    assert timeout_change["action"] == "increase"
    assert timeout_change["value_delta"] == 30
    assert timeout_change["current_value"] == 60
    assert timeout_change["proposed_value"] == 90
    
    # パッチ候補が保存されたことを確認
    assert mock_run_context.save_json.called
    call_args = mock_run_context.save_json.call_args
    assert call_args[0][0] == "patch_candidate_self_healing.json"
    assert call_args[0][1] == patch_candidate


@pytest.mark.asyncio
async def test_build_patch_candidate_trap_recovery_failed_timeout_increase(
    patch_agent,
    mock_run_context,
    site_config,
):
    """trap_recovery_failed の場合、timeout_sec を +30 秒するパッチ候補が生成されることを確認"""
    failure_context = {
        "site": "TEST_SITE",
        "run_id": "test_run_123",
        "error_type": "trap_recovery_failed",
        "error_class": "TrapPageDetected",
        "error_message": "Trap page detected but recovery failed",
    }
    
    failure_analysis = {
        "summary": "Trap recovery failed",
        "root_causes": ["Recovery timeout"],
        "suggested_fixes": [],
        "confidence": 0.7,
    }
    
    patch_candidate = await patch_agent.build_patch_candidate(
        failure_context=failure_context,
        failure_analysis=failure_analysis,
        site_config=site_config,
        run_context=mock_run_context,
    )
    
    assert patch_candidate is not None
    
    # timeout_sec の increase 提案が含まれていることを確認
    timeout_change = next(
        (c for c in patch_candidate["changes"] if c.get("path") == "discovery_settings.timeout_sec"),
        None
    )
    assert timeout_change is not None
    assert timeout_change["action"] == "increase"


@pytest.mark.asyncio
async def test_build_patch_candidate_selector_not_found_review_required(
    patch_agent,
    mock_run_context,
    site_config,
):
    """error_message に "selector" が含まれる場合、pdp selector に対する「要再確認」フラグが含まれることを確認"""
    failure_context = {
        "site": "TEST_SITE",
        "run_id": "test_run_123",
        "error_type": "pdp_extraction_failed",
        "error_class": "ValueError",
        "error_message": "Selector not found: .product-price",
    }
    
    failure_analysis = {
        "summary": "Selector not found",
        "root_causes": ["Selector may be outdated"],
        "suggested_fixes": ["Review selectors"],
        "confidence": 0.6,
    }
    
    patch_candidate = await patch_agent.build_patch_candidate(
        failure_context=failure_context,
        failure_analysis=failure_analysis,
        site_config=site_config,
        run_context=mock_run_context,
    )
    
    assert patch_candidate is not None
    
    # pdp selector に対する「要再確認」フラグが含まれていることを確認
    selector_change = next(
        (c for c in patch_candidate["changes"] if c.get("path") == "selectors.pdp"),
        None
    )
    assert selector_change is not None
    assert selector_change["action"] == "review_required"
    assert selector_change["reason"] == "Selector may be outdated or incorrect"
    assert "current_selectors" in selector_change


@pytest.mark.asyncio
async def test_build_patch_candidate_suggested_fixes_timeout(
    patch_agent,
    mock_run_context,
    site_config,
):
    """suggested_fixes に "timeout" が含まれる場合、timeout_sec を +30 秒するパッチ候補が生成されることを確認"""
    failure_context = {
        "site": "TEST_SITE",
        "run_id": "test_run_123",
        "error_type": "pdp_extraction_failed",
        "error_class": "TimeoutError",
        "error_message": "Extraction timeout",
    }
    
    failure_analysis = {
        "summary": "Extraction timeout",
        "root_causes": ["Timeout occurred"],
        "suggested_fixes": ["Increase timeout_sec to 90"],
        "confidence": 0.7,
    }
    
    patch_candidate = await patch_agent.build_patch_candidate(
        failure_context=failure_context,
        failure_analysis=failure_analysis,
        site_config=site_config,
        run_context=mock_run_context,
    )
    
    assert patch_candidate is not None
    
    # timeout_sec の increase 提案が含まれていることを確認
    timeout_change = next(
        (c for c in patch_candidate["changes"] if c.get("path") == "discovery_settings.timeout_sec"),
        None
    )
    assert timeout_change is not None
    assert timeout_change["action"] == "increase"


@pytest.mark.asyncio
async def test_build_patch_candidate_exception_handling(
    patch_agent,
    mock_run_context,
    site_config,
):
    """例外発生時でも None を返し、メインフローに影響しないことを確認"""
    # save_json が例外を投げるように設定
    mock_run_context.save_json.side_effect = Exception("Save failed")
    
    failure_context = {
        "site": "TEST_SITE",
        "run_id": "test_run_123",
        "error_type": "navigation_failed",
        "error_class": "TimeoutError",
        "error_message": "Navigation timeout",
    }
    
    failure_analysis = {
        "summary": "Navigation timeout",
        "root_causes": [],
        "suggested_fixes": [],
        "confidence": 0.8,
    }
    
    # 例外が発生しても None を返すことを確認
    patch_candidate = await patch_agent.build_patch_candidate(
        failure_context=failure_context,
        failure_analysis=failure_analysis,
        site_config=site_config,
        run_context=mock_run_context,
    )
    
    assert patch_candidate is None


@pytest.mark.asyncio
async def test_orchestrator_calls_patch_agent_on_failure(
    mock_run_context,
    site_config,
):
    """BrowserOrchestrator が失敗時に SelfHealingPatchAgent を呼び出すことを確認"""
    # SelfHealingPatchAgent のモック
    mock_patch_agent = AsyncMock()
    mock_patch_agent.build_patch_candidate = AsyncMock(
        return_value={
            "target_site": "TEST_SITE",
            "run_id": "test_run_123",
            "changes": [{"path": "discovery_settings.timeout_sec", "action": "increase"}],
        }
    )
    
    # BrowserOrchestrator を作成（patch_agent を注入）
    orchestrator = BrowserOrchestrator(
        runtime_kwargs={},
        patch_agent=mock_patch_agent,
        log=MagicMock(),
    )
    
    # FailureAnalysisAgent のモック
    mock_analysis_agent = AsyncMock()
    mock_analysis_agent.analyze_failure_context = AsyncMock(
        return_value={
            "summary": "Test analysis",
            "root_causes": [],
            "suggested_fixes": [],
            "confidence": 0.8,
        }
    )
    orchestrator.analysis_agent = mock_analysis_agent
    
    # BrowserExtractionService のモック（ValueError を投げる）
    with patch("app.agents.browser_orchestrator.BrowserExtractionService") as mock_extraction_service_class, \
         patch("app.agents.browser_orchestrator.TelemetryClient") as mock_telemetry_class:
        
        mock_extraction_service = AsyncMock()
        mock_extraction_service.extract_single_pdp = AsyncMock(
            side_effect=ValueError("Price not found on PDP.")
        )
        mock_extraction_service_class.return_value = mock_extraction_service
        
        mock_telemetry = AsyncMock()
        mock_telemetry.record_plp_state = AsyncMock()
        mock_telemetry_class.return_value = mock_telemetry
        
        # モック Page と Context
        mock_page = AsyncMock()
        mock_page.url = "https://example.com/product/1"
        mock_page.is_closed.return_value = False
        
        mock_context = AsyncMock()
        
        # run_pdp を実行
        result = await orchestrator.run_pdp(
            page=mock_page,
            context=mock_context,
            site="TEST_SITE",
            query="test query",
            site_config=site_config,
            settings={"timeout_sec": 60},
            run_context=mock_run_context,
            target_url="https://example.com/product/1",
        )
        
        # SelfHealingPatchAgent が呼ばれたことを確認
        assert mock_patch_agent.build_patch_candidate.called
        
        # self_healing_patch_candidate が evidence に含まれていることを確認
        assert result.ok is False
        assert "failure_context" in result.evidence
        assert "failure_analysis" in result.evidence
        assert "self_healing_patch_candidate" in result.evidence
        
        patch_candidate = result.evidence["self_healing_patch_candidate"]
        assert patch_candidate["target_site"] == "TEST_SITE"
        assert len(patch_candidate["changes"]) > 0


@pytest.mark.asyncio
async def test_orchestrator_no_patch_agent_does_not_break_flow(
    mock_run_context,
    site_config,
):
    """patch_agent が None の場合でも、メインフローが正常に動作することを確認"""
    # BrowserOrchestrator を作成（patch_agent を None に設定）
    # 注意: BrowserOrchestrator.__init__ は patch_agent が None の場合でも
    # SelfHealingPatchAgent を自動的に作成する可能性があるため、
    # 明示的に None を設定する必要がある
    orchestrator = BrowserOrchestrator(
        runtime_kwargs={},
        patch_agent=None,
        log=MagicMock(),
    )
    # __init__ 後に明示的に None を設定（自動生成を防ぐため）
    orchestrator.patch_agent = None
    
    # FailureAnalysisAgent のモック
    mock_analysis_agent = AsyncMock()
    mock_analysis_agent.analyze_failure_context = AsyncMock(
        return_value={
            "summary": "Test analysis",
            "root_causes": [],
            "suggested_fixes": [],
            "confidence": 0.8,
        }
    )
    orchestrator.analysis_agent = mock_analysis_agent
    
    # BrowserExtractionService のモック（ValueError を投げる）
    with patch("app.agents.browser_orchestrator.BrowserExtractionService") as mock_extraction_service_class, \
         patch("app.agents.browser_orchestrator.TelemetryClient") as mock_telemetry_class:
        
        mock_extraction_service = AsyncMock()
        mock_extraction_service.extract_single_pdp = AsyncMock(
            side_effect=ValueError("Price not found on PDP.")
        )
        mock_extraction_service_class.return_value = mock_extraction_service
        
        mock_telemetry = AsyncMock()
        mock_telemetry.record_plp_state = AsyncMock()
        mock_telemetry_class.return_value = mock_telemetry
        
        # モック Page と Context
        mock_page = AsyncMock()
        mock_page.url = "https://example.com/product/1"
        mock_page.is_closed.return_value = False
        
        mock_context = AsyncMock()
        
        # run_pdp を実行
        result = await orchestrator.run_pdp(
            page=mock_page,
            context=mock_context,
            site="TEST_SITE",
            query="test query",
            site_config=site_config,
            settings={"timeout_sec": 60},
            run_context=mock_run_context,
            target_url="https://example.com/product/1",
        )
        
        # self_healing_patch_candidate が evidence に含まれていないことを確認
        assert result.ok is False
        assert "failure_context" in result.evidence
        assert "failure_analysis" in result.evidence
        # patch_agent が None のため、self_healing_patch_candidate は含まれない
        assert "self_healing_patch_candidate" not in result.evidence or result.evidence.get("self_healing_patch_candidate") is None

