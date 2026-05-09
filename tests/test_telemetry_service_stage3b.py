"""
TelemetryService Stage 3B Step 1 の動作確認テスト

基本的なインポート、インスタンス化、メソッドの動作を確認します。
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.browser.telemetry import (
    FailureContext,
    RunPhase,
    TelemetryService,
)
from app.core.run_context import RunContext


def test_imports():
    """インポートの確認"""
    assert TelemetryService is not None
    assert RunPhase is not None
    assert FailureContext is not None


def test_run_phase_enum():
    """RunPhase Enum の確認"""
    assert RunPhase.PLP_DISCOVERY == "plp_discovery"
    assert RunPhase.PDP_EXTRACT == "pdp_extract"
    assert RunPhase.RECOVERY == "recovery"
    assert RunPhase.LEARNING == "learning"
    assert RunPhase.MATERIALIZE == "materialize"

    # Enum値の確認
    assert RunPhase.PLP_DISCOVERY.value == "plp_discovery"
    assert RunPhase.PDP_EXTRACT.value == "pdp_extract"


def test_failure_context():
    """FailureContext dataclass の作成確認"""
    failure = FailureContext(
        site_code="MONCLER_OFFICIAL",
        url="https://www.moncler.com/en-int/men/",
        phase=RunPhase.PLP_DISCOVERY,
        exception=ValueError("Test error"),
        retry_count=1,
        query="GUCCI",
    )

    assert failure.site_code == "MONCLER_OFFICIAL"
    assert failure.url == "https://www.moncler.com/en-int/men/"
    assert failure.phase == RunPhase.PLP_DISCOVERY
    assert isinstance(failure.exception, ValueError)
    assert failure.retry_count == 1
    assert failure.query == "GUCCI"

    # オプショナルフィールドのデフォルト値確認
    failure2 = FailureContext(
        site_code="TEST",
        url="https://example.com",
        phase=RunPhase.PDP_EXTRACT,
    )
    assert failure2.exception is None
    assert failure2.retry_count == 0
    assert failure2.query is None
    assert failure2.site_config is None


def test_telemetry_service_init(tmp_path):
    """TelemetryService の初期化確認"""
    run_context = RunContext(base_path=str(tmp_path / "runs"))
    telemetry = TelemetryService(run_context=run_context)

    assert telemetry.run_context == run_context
    assert telemetry.logger is not None

    # カスタムロガーの指定
    import logging

    custom_logger = logging.getLogger("custom_test")
    telemetry2 = TelemetryService(run_context=run_context, logger=custom_logger)
    assert telemetry2.logger == custom_logger


@pytest.mark.asyncio
async def test_record_plp_state_basic(tmp_path):
    """record_plp_state の基本的な動作確認"""
    run_context = RunContext(base_path=str(tmp_path / "runs"))
    telemetry = TelemetryService(run_context=run_context)

    # モック Page オブジェクト
    mock_page = MagicMock()
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.content = AsyncMock(return_value="<html><body>Test</body></html>")
    mock_page.locator = MagicMock()

    # locator().count() のモック
    mock_locator = MagicMock()
    mock_locator.count = AsyncMock(return_value=5)
    mock_page.locator.return_value = mock_locator

    # take_screenshot のモック
    run_context.take_screenshot = AsyncMock(return_value="screenshot_path.png")

    # 実行
    await telemetry.record_plp_state(
        mock_page,
        name="test_plp",
        selectors=["div.product", "a.link"],
    )

    # 確認: DOM保存が呼ばれたか
    assert mock_page.content.called

    # 確認: セレクタカウントが呼ばれたか
    assert mock_page.locator.called

    # 確認: スクリーンショットが呼ばれたか
    assert run_context.take_screenshot.called


@pytest.mark.asyncio
async def test_record_plp_state_with_site_config(tmp_path):
    """record_plp_state で site_config から selectors を自動取得する確認"""
    run_context = RunContext(base_path=str(tmp_path / "runs"))
    telemetry = TelemetryService(run_context=run_context)

    mock_page = MagicMock()
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.content = AsyncMock(return_value="<html></html>")
    mock_page.locator = MagicMock()

    mock_locator = MagicMock()
    mock_locator.count = AsyncMock(return_value=0)
    mock_page.locator.return_value = mock_locator

    site_config = {
        "selectors": {
            "pdp": {
                "pdp_link_selectors": ["a.product-link"],
                "plp_container_selectors": ["div.products"],
            }
        }
    }

    await telemetry.record_plp_state(
        mock_page,
        name="test_plp",
        site_config=site_config,
    )

    # site_configからselectorsが自動取得され、locatorが呼ばれたことを確認
    assert mock_page.locator.called


@pytest.mark.asyncio
async def test_record_plp_state_closed_page(tmp_path):
    """record_plp_state で page が閉じている場合の動作確認"""
    run_context = RunContext(base_path=str(tmp_path / "runs"))
    telemetry = TelemetryService(run_context=run_context)

    mock_page = MagicMock()
    mock_page.is_closed = MagicMock(return_value=True)

    # 閉じたページでもエラーにならないことを確認
    await telemetry.record_plp_state(mock_page, name="test_plp")

    # content() は呼ばれない
    assert not hasattr(mock_page, "content") or not mock_page.content.called


@pytest.mark.asyncio
async def test_record_success(tmp_path):
    """record_success の基本的な動作確認"""
    run_context = RunContext(base_path=str(tmp_path / "runs"))
    telemetry = TelemetryService(run_context=run_context)

    # 実行
    await telemetry.record_success(
        phase=RunPhase.PLP_DISCOVERY,
        url="https://www.moncler.com/en-int/men/",
    )

    # JSONファイルが作成されたか確認
    success_files = list(run_context.run_path.glob("success_*.json"))
    assert len(success_files) > 0

    # ファイル内容の確認
    import json

    with open(success_files[0], encoding="utf-8") as f:
        data = json.load(f)
        assert data["phase"] == "plp_discovery"
        assert data["url"] == "https://www.moncler.com/en-int/men/"
        assert "timestamp" in data


@pytest.mark.asyncio
async def test_record_success_with_result(tmp_path):
    """record_success で DiscoveryResult を含む場合の確認"""
    run_context = RunContext(base_path=str(tmp_path / "runs"))
    telemetry = TelemetryService(run_context=run_context)

    # モック DiscoveryResult
    mock_result = MagicMock()
    mock_result.ok = True
    mock_result.site = "MONCLER_OFFICIAL"
    mock_result.query = "GUCCI"
    mock_result.message = "Success"

    await telemetry.record_success(
        phase=RunPhase.PDP_EXTRACT,
        result=mock_result,
    )

    # JSONファイルの確認
    success_files = list(run_context.run_path.glob("success_*.json"))
    assert len(success_files) > 0

    import json

    with open(success_files[0], encoding="utf-8") as f:
        data = json.load(f)
        assert data["phase"] == "pdp_extract"
        assert "result" in data
        assert data["result"]["ok"] is True
        assert data["result"]["site"] == "MONCLER_OFFICIAL"


@pytest.mark.asyncio
async def test_record_failure(tmp_path):
    """record_failure の基本的な動作確認"""
    run_context = RunContext(base_path=str(tmp_path / "runs"))
    telemetry = TelemetryService(run_context=run_context)

    # モック Page オブジェクト
    mock_page = MagicMock()
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.content = AsyncMock(return_value="<html><body>Error</body></html>")
    mock_page.locator = MagicMock()

    mock_locator = MagicMock()
    mock_locator.count = AsyncMock(return_value=0)
    mock_page.locator.return_value = mock_locator

    # take_screenshot のモック
    run_context.take_screenshot = AsyncMock(return_value="failure_screenshot.png")

    failure = FailureContext(
        site_code="MONCLER_OFFICIAL",
        url="https://www.moncler.com/en-int/men/",
        phase=RunPhase.PLP_DISCOVERY,
        exception=ValueError("Test error"),
        retry_count=1,
        query="GUCCI",
        site_config={
            "selectors": {
                "pdp": {
                    "title_selectors": ["h1.title"],
                    "price_selectors": ["span.price"],
                }
            }
        },
    )

    await telemetry.record_failure(failure, page=mock_page)

    # DOM保存が呼ばれたか
    assert mock_page.content.called

    # スクリーンショットが呼ばれたか
    assert run_context.take_screenshot.called

    # fail_snapshot.md が作成されたか確認
    snapshot_file = run_context.run_path / "fail_snapshot.md"
    assert snapshot_file.exists()

    # ファイル内容の確認
    content = snapshot_file.read_text(encoding="utf-8")
    assert "MONCLER_OFFICIAL" in content
    assert "PLP_DISCOVERY" in content or "plp_discovery" in content
    assert "GUCCI" in content


@pytest.mark.asyncio
async def test_record_failure_without_page(tmp_path):
    """record_failure で page が None の場合の動作確認"""
    run_context = RunContext(base_path=str(tmp_path / "runs"))
    telemetry = TelemetryService(run_context=run_context)

    failure = FailureContext(
        site_code="TEST",
        url="https://example.com",
        phase=RunPhase.PDP_EXTRACT,
        exception=ValueError("Test"),
    )

    # page が None でもエラーにならないことを確認
    await telemetry.record_failure(failure, page=None)

    # fail_snapshot.md は作成される
    snapshot_file = run_context.run_path / "fail_snapshot.md"
    assert snapshot_file.exists()


@pytest.mark.asyncio
async def test_internal_methods(tmp_path):
    """内部メソッドの動作確認"""
    run_context = RunContext(base_path=str(tmp_path / "runs"))
    telemetry = TelemetryService(run_context=run_context)

    # _save_dom の確認
    mock_page = MagicMock()
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.content = AsyncMock(return_value="<html></html>")

    await telemetry._save_dom(mock_page, "test_dom")

    # DOMファイルが作成されたか確認
    dom_file = run_context.run_path / "test_dom.html"
    assert dom_file.exists()
    assert dom_file.read_text(encoding="utf-8") == "<html></html>"

    # _save_json の確認
    test_data = {"key": "value", "number": 42}
    await telemetry._save_json(test_data, "test_json")

    # JSONファイルが作成されたか確認
    json_file = run_context.run_path / "test_json.json"
    assert json_file.exists()

    import json

    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)
        assert data == test_data

    # _count_selectors の確認
    mock_page2 = MagicMock()
    mock_page2.is_closed = MagicMock(return_value=False)
    mock_page2.locator = MagicMock()

    mock_locator = MagicMock()
    mock_locator.count = AsyncMock(return_value=3)
    mock_page2.locator.return_value = mock_locator

    await telemetry._count_selectors(mock_page2, ["div.test"], name="test_counts")

    # カウントJSONファイルが作成されたか確認
    counts_file = run_context.run_path / "test_counts.json"
    assert counts_file.exists()

    with open(counts_file, encoding="utf-8") as f:
        data = json.load(f)
        assert "selector_counts" in data
        assert data["selector_counts"]["div.test"] == 3


@pytest.mark.asyncio
async def test_record_raw_hrefs(tmp_path):
    """record_raw_hrefs の動作確認"""
    run_context = RunContext(base_path=str(tmp_path / "runs"))
    telemetry = TelemetryService(run_context=run_context)

    # テスト用のURLリスト
    hrefs = [
        "https://example.com/product1",
        "https://example.com/product2",
        "https://example.com/product3",
    ] * 100  # 300個のURL（limitを超える）

    await telemetry.record_raw_hrefs(hrefs, name="test_hrefs", limit=200)

    # JSONファイルが作成されたか確認
    hrefs_file = run_context.run_path / "test_hrefs.json"
    assert hrefs_file.exists()

    import json

    with open(hrefs_file, encoding="utf-8") as f:
        data = json.load(f)
        assert "count" in data
        assert "raw_hrefs" in data
        assert data["count"] == 300  # 元の数
        assert len(data["raw_hrefs"]) == 200  # limitで制限された数
        assert data["raw_hrefs"][0] == "https://example.com/product1"


@pytest.mark.asyncio
async def test_record_raw_hrefs_empty_list(tmp_path):
    """record_raw_hrefs で空のリストを渡した場合の確認"""
    run_context = RunContext(base_path=str(tmp_path / "runs"))
    telemetry = TelemetryService(run_context=run_context)

    await telemetry.record_raw_hrefs([], name="empty_hrefs")

    hrefs_file = run_context.run_path / "empty_hrefs.json"
    assert hrefs_file.exists()

    import json

    with open(hrefs_file, encoding="utf-8") as f:
        data = json.load(f)
        assert data["count"] == 0
        assert len(data["raw_hrefs"]) == 0
