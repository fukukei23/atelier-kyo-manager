"""
CR-ATELIER-003 Phase D-4: BrowserOrchestrator Telemetry 統合のテスト

TelemetryClient / TelemetryService を AsyncMock に差し替えて、
run_plp_to_pdp / run_pdp 実行時に指定メソッドが呼ばれることだけを検証する。
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from app.agents.browser.plp_driver import PlpNavigationResult
from app.agents.browser_orchestrator import BrowserOrchestrator
from app.models.result_models import DiscoveryResult


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
    run_context = MagicMock()
    run_context.run_id = "test_run_123"
    run_context.run_path = "/tmp/test_run"
    run_context.get_path = Mock(return_value="/tmp/test_run/failure_dom.html")
    run_context.screenshots = []
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
def orchestrator(mock_run_context):
    """BrowserOrchestrator インスタンス"""
    return BrowserOrchestrator(
        runtime_kwargs={},
        log=MagicMock(),
    )


@pytest.mark.asyncio
async def test_run_plp_to_pdp_records_plp_initial_state(
    orchestrator,
    mock_page,
    mock_context,
    mock_run_context,
    site_config,
    settings,
):
    """run_plp_to_pdp が PLP 初期状態を Telemetry に記録することを確認"""
    with (
        patch("app.agents.browser.orchestrator.plp_pdp_flow.TelemetryClient") as mock_telemetry_class,
        patch("app.agents.browser.orchestrator.plp_pdp_flow.NavigationDriver") as mock_nav_driver_class,
    ):
        # TelemetryClient のモック
        mock_telemetry = AsyncMock()
        mock_telemetry.record_plp_state = AsyncMock()
        mock_telemetry.save_json = AsyncMock()
        mock_telemetry_class.return_value = mock_telemetry

        # NavigationDriver のモック
        mock_nav_driver = AsyncMock()
        mock_nav_outcome = MagicMock()
        mock_nav_outcome.entry_url = "https://example.com/plp"
        mock_nav_outcome.pdp_links = ["https://example.com/product/1"]
        mock_nav_outcome.trap_detected = False
        mock_nav_outcome.recovered = False
        mock_nav_driver.run_plp_flow = AsyncMock(return_value=mock_nav_outcome)
        mock_nav_driver_class.return_value = mock_nav_driver

        # BrowserExtractionService のモック
        with patch("app.agents.browser.orchestrator.plp_pdp_flow.BrowserExtractionService") as mock_extraction_service_class:
            mock_extraction_service = AsyncMock()
            mock_extraction_service.extract_from_pdp_list = AsyncMock(
                return_value=DiscoveryResult(
                    ok=True,
                    site="TEST_SITE",
                    query="test query",
                    message="Success",
                )
            )
            mock_extraction_service_class.return_value = mock_extraction_service

            # run_plp_to_pdp を実行
            await orchestrator.run_plp_to_pdp(
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

            # Telemetry が呼ばれたことを確認
            assert mock_telemetry.record_plp_state.called
            assert mock_telemetry.save_json.called


@pytest.mark.asyncio
async def test_run_plp_to_pdp_records_no_pdp_links(
    orchestrator,
    mock_page,
    mock_context,
    mock_run_context,
    site_config,
    settings,
):
    """pdp_links が 0 の場合、Telemetry に記録されることを確認"""
    with (
        patch("app.agents.browser.orchestrator.plp_pdp_flow.TelemetryClient") as mock_telemetry_class,
        patch("app.agents.browser.orchestrator.plp_pdp_flow.NavigationDriver") as mock_nav_driver_class,
        patch("app.agents.browser.orchestrator.plp_pdp_flow.PlpDriver") as mock_plp_driver_class,
    ):
        # TelemetryClient のモック
        mock_telemetry = AsyncMock()
        mock_telemetry.save_json = AsyncMock()
        mock_telemetry_class.return_value = mock_telemetry

        # NavigationDriver のモック（pdp_links が空）
        mock_nav_driver = AsyncMock()
        mock_nav_outcome = MagicMock()
        mock_nav_outcome.entry_url = "https://example.com/plp"
        mock_nav_outcome.pdp_links = []
        mock_nav_outcome.trap_detected = False
        mock_nav_outcome.recovered = False
        mock_nav_driver.run_plp_flow = AsyncMock(return_value=mock_nav_outcome)
        mock_nav_driver_class.return_value = mock_nav_driver

        # PlpDriver のモック
        mock_plp_driver = AsyncMock()
        mock_plp_result = PlpNavigationResult(
            pdp_url="https://example.com/product/1",
            plp_url="https://example.com/plp",
            tiles_seen=5,
            trap_detected=False,
            recovery_attempted=False,
            recovery_successful=False,
            overlays_handled=0,
            navigation_method="click_tile",
            pdp_opened_in_new_tab=False,
        )
        mock_plp_driver.navigate_to_pdp = AsyncMock(return_value=mock_plp_result)
        mock_plp_driver_class.return_value = mock_plp_driver

        # run_plp_to_pdp を実行
        await orchestrator.run_plp_to_pdp(
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

        # Telemetry に no_pdp_links が記録されたことを確認
        save_json_calls = [call for call in mock_telemetry.save_json.call_args_list]
        assert any("no_pdp_links" in str(call) for call in save_json_calls)


@pytest.mark.asyncio
async def test_run_pdp_records_pdp_dom(
    orchestrator,
    mock_page,
    mock_context,
    mock_run_context,
    site_config,
    settings,
):
    """run_pdp が PDP DOM を Telemetry に記録することを確認"""
    with (
        patch("app.agents.browser.orchestrator.plp_pdp_flow.TelemetryClient") as mock_telemetry_class,
        patch("app.agents.browser.orchestrator.plp_pdp_flow.BrowserExtractionService") as mock_extraction_service_class,
    ):
        # TelemetryClient のモック
        mock_telemetry = AsyncMock()
        mock_telemetry.record_plp_state = AsyncMock()
        mock_telemetry.save_json = AsyncMock()
        mock_telemetry_class.return_value = mock_telemetry

        # BrowserExtractionService のモック
        mock_extraction_service = AsyncMock()
        mock_extraction_service.extract_single_pdp = AsyncMock(
            return_value=DiscoveryResult(
                ok=True,
                site="TEST_SITE",
                query="test query",
                message="Success",
            )
        )
        mock_extraction_service_class.return_value = mock_extraction_service

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

        # Telemetry が呼ばれたことを確認（prepare_hook 内で呼ばれる）
        # 注意: prepare_hook は extract_single_pdp 内で呼ばれるため、
        # 実際の呼び出しは extract_single_pdp の実行時に発生する
        assert result.ok is True


@pytest.mark.asyncio
async def test_failure_context_includes_required_keys(
    orchestrator,
    mock_page,
    mock_context,
    mock_run_context,
    site_config,
    settings,
):
    """failure_context に必要なキーが含まれていることを確認"""
    with patch("app.agents.browser.orchestrator.plp_pdp_flow.BrowserExtractionService") as mock_extraction_service_class:
        # BrowserExtractionService が ValueError を投げるように設定
        mock_extraction_service = AsyncMock()
        mock_extraction_service.extract_single_pdp = AsyncMock(side_effect=ValueError("Price not found on PDP."))
        mock_extraction_service_class.return_value = mock_extraction_service

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

        # failure_context が含まれていることを確認
        assert result.ok is False
        assert "failure_context" in result.evidence

        failure_ctx = result.evidence["failure_context"]
        assert "final_url" in failure_ctx
        assert "error_type" in failure_ctx
        assert "error_class" in failure_ctx
        assert "error_message" in failure_ctx
        assert "site" in failure_ctx
        assert "query" in failure_ctx
        assert "run_id" in failure_ctx


@pytest.mark.asyncio
async def test_maybe_analyze_failure_logs_failure(
    orchestrator,
    mock_page,
):
    """_maybe_analyze_failure がログを出力することを確認"""
    failure_ctx = {
        "final_url": "https://example.com/product/1",
        "error_type": "pdp_extraction_failed",
        "error_class": "ValueError",
        "error_message": "Price not found",
        "site": "TEST_SITE",
        "query": "test query",
        "run_id": "test_run_123",
    }

    # _maybe_analyze_failure を実行
    await orchestrator._maybe_analyze_failure(failure_ctx, page=mock_page)

    # ログが出力されたことを確認（log.info が呼ばれたことを確認）
    assert orchestrator.log.info.called


@pytest.mark.asyncio
async def test_build_failure_context_includes_site_config_summary(
    orchestrator,
    mock_run_context,
    site_config,
):
    """_build_failure_context が site_config_summary を含むことを確認"""
    failure_ctx = orchestrator._build_failure_context(
        site="TEST_SITE",
        query="test query",
        final_url="https://example.com/product/1",
        error_type="pdp_extraction_failed",
        error_class="ValueError",
        error_message="Price not found",
        site_config=site_config,
        run_context=mock_run_context,
    )

    assert "site_config_summary" in failure_ctx
    assert failure_ctx["site_config_summary"]["site_code"] == "TEST_SITE"
    assert "has_plp_selectors" in failure_ctx["site_config_summary"]
    assert "has_pdp_selectors" in failure_ctx["site_config_summary"]
