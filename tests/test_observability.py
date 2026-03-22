# -*- coding: utf-8 -*-
"""
test_observability.py
======================================================================
Observabilityユーティリティのユニットテスト
======================================================================
"""
import pytest
import time
import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

# プロジェクトルートをパスに追加
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


class MockRunContext:
    """RunContextのモック"""
    def __init__(self):
        self.run_id = "test_run_001"
        self.output_dir = APP_ROOT / "output"
        self.session_id = "test_session"
        self._saved_content = {}
        self._saved_json = {}

    async def take_screenshot(self, page, name):
        return f"{name}.png"

    def save_content(self, filename, content):
        self._saved_content[filename] = content

    async def save_content_async(self, filename, content):
        self._saved_content[filename] = content

    def save_json(self, filename, data):
        self._saved_json[filename] = data

    @property
    def logger(self):
        return MagicMock()


@pytest.mark.asyncio
async def test_save_dom_handles_closed_page():
    """閉じたページでsave_domがエラーなく動作することを確認"""
    from app.utils.observability import save_dom

    mock_page = MagicMock()
    mock_page.is_closed.return_value = True

    ctx = MockRunContext()
    await save_dom(ctx, mock_page, "test")

    # 何も保存されていないことを確認
    assert len(ctx._saved_content) == 0


@pytest.mark.asyncio
async def test_save_dom_saves_html():
    """save_domがHTMLを保存することを確認"""
    from app.utils.observability import save_dom

    mock_page = MagicMock()
    mock_page.is_closed.return_value = False
    mock_page.content = AsyncMock(return_value="<html><body>Test</body></html>")

    ctx = MockRunContext()
    await save_dom(ctx, mock_page, "test")

    assert "test.html" in ctx._saved_content
    assert "<html><body>Test</body></html>" in ctx._saved_content["test.html"]


@pytest.mark.asyncio
async def test_count_selectors_returns_counts():
    """count_selectorsが正しいカウントを返すことを確認"""
    from app.utils.observability import count_selectors

    mock_page = MagicMock()
    mock_page.is_closed.return_value = False
    mock_page.locator = MagicMock()

    # モックのlocatorをに設定
    locator_mock = MagicMock()
    locator_mock.count = AsyncMock(return_value=5)
    mock_page.locator.return_value = locator_mock

    ctx = MockRunContext()
    await count_selectors(ctx, mock_page, ["div.product-item"])

    assert "selector_counts.json" in ctx._saved_json
    assert ctx._saved_json["selector_counts.json"]["selector_counts"]["div.product-item"] == 5


@pytest.mark.asyncio
async def test_log_operation_metric():
    """log_operation_metricが正しく動作することを確認"""
    from app.utils.observability import log_operation_metric

    ctx = MockRunContext()
    start = time.time()

    await log_operation_metric(
        ctx,
        "test_operation",
        (time.time() - start) * 1000,
        success=True,
        details={"key": "value"}
    )

    # メトリクスが保存されていることを確認
    metric_files = [k for k in ctx._saved_json.keys() if k.startswith("metric_test_operation")]
    assert len(metric_files) > 0

    metric = ctx._saved_json[metric_files[0]]
    assert metric["operation"] == "test_operation"
    assert metric["success"] is True
    assert metric["details"] == {"key": "value"}


@pytest.mark.asyncio
async def test_save_raw_hrefs():
    """save_raw_hrefsがURLリストを保存することを確認"""
    from app.utils.observability import save_raw_hrefs

    ctx = MockRunContext()
    hrefs = ["https://example.com/1", "https://example.com/2", "https://example.com/3"]

    await save_raw_hrefs(ctx, hrefs, name="test_hrefs")

    assert "test_hrefs.json" in ctx._saved_json
    assert ctx._saved_json["test_hrefs.json"]["count"] == 3
    assert len(ctx._saved_json["test_hrefs.json"]["raw_hrefs"]) == 3


@pytest.mark.asyncio
async def test_maybe_await_with_sync():
    """_maybe_awaitが同期的値を正しく処理することを確認"""
    from app.utils.observability import _maybe_await

    result = await _maybe_await("sync_value")
    assert result == "sync_value"


@pytest.mark.asyncio
async def test_maybe_await_with_async():
    """_maybe_awaitが非同期的値を正しく処理することを確認"""
    from app.utils.observability import _maybe_await

    async def async_value():
        return "async_value"

    result = await _maybe_await(async_value())
    assert result == "async_value"
