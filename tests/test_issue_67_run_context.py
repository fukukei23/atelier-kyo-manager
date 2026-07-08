"""Issue #67: run_context.py カバレッジ 43→90%+

Tests for uncovered paths:
- L33-40: save_json (success + error)
- L42-43: get_path
- L45-51: append_log (success + error)
- L53-70: take_screenshot async (page=None, page closed, success, exception)
- L72-77: setup_system_logger
- L80-89: save_content (success + error)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.run_context import RunContext


@pytest.fixture()
def ctx(tmp_path):
    return RunContext(base_path=str(tmp_path / "runs"))


class TestSaveJson:
    def test_save_json_writes_file(self, ctx):
        data = {"key": "value", "num": 42}
        ctx.save_json("test.json", data)
        path = ctx.get_path("test.json")
        assert path.exists()
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded == data

    def test_save_json_handles_error(self, ctx):
        with patch("builtins.open", side_effect=PermissionError("no access")):
            ctx.save_json("bad.json", {"a": 1})


class TestGetPath:
    def test_get_path_returns_expected(self, ctx):
        result = ctx.get_path("report.html")
        assert isinstance(result, Path)
        assert result.name == "report.html"
        assert str(ctx.run_path) in str(result)


class TestAppendLog:
    def test_append_log_creates_file(self, ctx):
        ctx.append_log("log.txt", "first line")
        path = ctx.get_path("log.txt")
        assert path.exists()
        assert "first line" in path.read_text(encoding="utf-8")

    def test_append_log_appends(self, ctx):
        ctx.append_log("log.txt", "line1")
        ctx.append_log("log.txt", "line2")
        content = ctx.get_path("log.txt").read_text(encoding="utf-8")
        assert "line1" in content
        assert "line2" in content

    def test_append_log_handles_error(self, ctx):
        with patch("builtins.open", side_effect=PermissionError("no access")):
            ctx.append_log("bad.txt", "content")


class TestTakeScreenshot:
    @pytest.mark.asyncio
    async def test_none_page_returns_none(self, ctx):
        result = await ctx.take_screenshot(None, "test")
        assert result is None

    @pytest.mark.asyncio
    async def test_closed_page_returns_none(self, ctx):
        page = MagicMock()
        page.is_closed.return_value = True
        result = await ctx.take_screenshot(page, "test")
        assert result is None

    @pytest.mark.asyncio
    async def test_successful_screenshot(self, ctx):
        page = MagicMock()
        page.is_closed.return_value = False
        page.screenshot = AsyncMock()
        result = await ctx.take_screenshot(page, "homepage")
        assert result is not None
        assert "homepage" in result
        assert ctx.screenshot_counter == 1

    @pytest.mark.asyncio
    async def test_screenshot_exception_returns_none(self, ctx):
        page = AsyncMock()
        page.is_closed.return_value = False
        page.screenshot = AsyncMock(side_effect=RuntimeError("browser crashed"))
        result = await ctx.take_screenshot(page, "fail")
        assert result is None


class TestSetupSystemLogger:
    def test_returns_file_handler(self, ctx):
        handler = ctx.setup_system_logger()
        assert isinstance(handler, logging.FileHandler)
        log_path = ctx.get_path("system.log")
        assert log_path.parent.exists()
        handler.close()


class TestSaveContent:
    def test_save_content_writes_file(self, ctx):
        ctx.save_content("report.html", "<h1>Hello</h1>")
        path = ctx.get_path("report.html")
        assert path.exists()
        assert path.read_text(encoding="utf-8") == "<h1>Hello</h1>"

    def test_save_content_handles_error(self, ctx):
        with patch.object(Path, "write_text", side_effect=PermissionError("denied")):
            ctx.save_content("bad.html", "content")
