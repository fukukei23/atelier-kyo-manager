import asyncio
import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.run_context import RunContext


def test_init_creates_directories(tmp_path: Path):
    """__init__ が run_path と screenshots_path を作成"""
    base_path = tmp_path / "test_runs"
    rc = RunContext(base_path=str(base_path))

    assert rc.run_path.exists()
    assert rc.run_path.is_dir()
    assert rc.screenshots_path.exists()
    assert rc.screenshots_path.is_dir()
    assert rc.screenshots_path.parent == rc.run_path


def test_save_json_normal(tmp_path: Path):
    """save_json 正常系"""
    rc = RunContext(base_path=str(tmp_path))
    data = {"key": "value", "list": [1, 2, 3]}
    result = rc.save_json("test.json", data)

    assert result is None
    saved_path = rc.run_path / "test.json"
    assert saved_path.exists()
    with open(saved_path, "r", encoding="utf-8") as f:
        assert json.load(f) == data


def test_save_json_exception(tmp_path: Path):
    """save_json 例外時にエラーログ出力"""
    rc = RunContext(base_path=str(tmp_path))
    rc.run_path = tmp_path / "non_existent_dir"

    with patch.object(logging, "error") as mock_log:
        rc.save_json("fail.json", {"a": 1})
        mock_log.assert_called_once()
        assert "Failed to save JSON" in mock_log.call_args[0][0]


def test_get_path(tmp_path: Path):
    """get_path が run_path / filename を返す"""
    rc = RunContext(base_path=str(tmp_path))
    assert rc.get_path("myfile.txt") == rc.run_path / "myfile.txt"


def test_append_log_normal(tmp_path: Path):
    """append_log が正常にログを追記"""
    rc = RunContext(base_path=str(tmp_path))

    rc.append_log("log.txt", "Line 1")
    rc.append_log("log.txt", "Line 2")

    content = (rc.run_path / "log.txt").read_text(encoding="utf-8")
    assert content == "Line 1\nLine 2\n"


def test_append_log_exception(tmp_path: Path):
    """append_log 例外時にエラーログ出力"""
    rc = RunContext(base_path=str(tmp_path))

    with patch("builtins.open", side_effect=IOError("Permission denied")):
        with patch.object(logging, "error") as mock_log:
            rc.append_log("log.txt", "fail")
            mock_log.assert_called_once()
            assert "Failed to append to log" in mock_log.call_args[0][0]


def test_take_screenshot_none_page(tmp_path: Path):
    """page=None → None"""
    rc = RunContext(base_path=str(tmp_path))
    result = asyncio.run(rc.take_screenshot(None, "null"))
    assert result is None
    assert rc.screenshot_counter == 0


def test_take_screenshot_closed_page(tmp_path: Path):
    """page.is_closed()=True → None"""
    rc = RunContext(base_path=str(tmp_path))
    mock_page = MagicMock()
    mock_page.is_closed.return_value = True

    result = asyncio.run(rc.take_screenshot(mock_page, "closed"))
    assert result is None
    assert rc.screenshot_counter == 0


def test_take_screenshot_normal(tmp_path: Path):
    """正常スクリーンショット → パス返却"""
    rc = RunContext(base_path=str(tmp_path))
    mock_page = MagicMock()
    mock_page.is_closed.return_value = False
    mock_page.screenshot = AsyncMock()

    result = asyncio.run(rc.take_screenshot(mock_page, "success"))

    expected_path = str(rc.screenshots_path / "00_success.png")
    assert result == expected_path
    mock_page.screenshot.assert_awaited_once_with(path=expected_path)
    assert rc.screenshot_counter == 1


def test_take_screenshot_exception(tmp_path: Path):
    """スクリーンショット例外 → None"""
    rc = RunContext(base_path=str(tmp_path))
    mock_page = MagicMock()
    mock_page.is_closed.return_value = False
    mock_page.screenshot = AsyncMock(side_effect=Exception("Browser crashed"))

    with patch.object(logging, "error") as mock_log:
        result = asyncio.run(rc.take_screenshot(mock_page, "crash"))
        assert result is None
        assert rc.screenshot_counter == 0
        mock_log.assert_called_once()
        assert "Failed to take screenshot" in mock_log.call_args[0][0]


def test_setup_system_logger(tmp_path: Path):
    """FileHandler + Formatter を返す"""
    rc = RunContext(base_path=str(tmp_path))
    handler = rc.setup_system_logger()

    assert isinstance(handler, logging.FileHandler)
    assert isinstance(handler.formatter, logging.Formatter)
    assert handler.baseFilename == str(rc.get_path("system.log"))
    handler.close()


def test_save_content_normal(tmp_path: Path):
    """save_content 正常系"""
    rc = RunContext(base_path=str(tmp_path))
    rc.save_content("content.txt", "テスト")

    assert (rc.run_path / "content.txt").read_text(encoding="utf-8") == "テスト"


def test_save_content_exception(tmp_path: Path):
    """save_content 例外時にエラーログ出力"""
    rc = RunContext(base_path=str(tmp_path))

    with patch.object(Path, "write_text", side_effect=IOError("Disk full")):
        with patch.object(logging, "error") as mock_log:
            rc.save_content("fail.txt", "失敗")
            mock_log.assert_called_once()
            assert "Failed to save content" in mock_log.call_args[0][0]
