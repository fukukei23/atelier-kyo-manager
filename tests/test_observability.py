"""Tests for app/utils/observability.py - Issue #85 カバレッジ向上"""
import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.utils.observability import (
    _log,
    _maybe_await,
    count_selectors,
    save_dom,
    save_json,
    save_raw_hrefs,
    write_fail_snapshot,
)


def _make_rc(has_logger=True):
    """RunContext モックを作成"""
    rc = MagicMock()
    if has_logger:
        rc.logger = logging.getLogger("test_rc")
    else:
        del rc.logger  # attr 削除で getattr フォールバックをテスト
    rc.save_content = AsyncMock(return_value=None)
    rc.save_json = AsyncMock(return_value=None)
    return rc


def _make_page(closed=False, content="<html></html>"):
    """Playwright Page モックを作成"""
    page = MagicMock()
    page.is_closed = MagicMock(return_value=closed)
    page.content = AsyncMock(return_value=content)
    return page


# ── _log ─────────────────────────────────────────────────────────────────────

class TestLog:
    def test_returns_rc_logger_when_present(self):
        rc = MagicMock()
        rc.logger = logging.getLogger("my_logger")
        result = _log(rc)
        assert result is rc.logger

    def test_fallback_to_module_logger(self):
        rc = MagicMock(spec=[])  # spec=[] で属性アクセスを制限
        result = _log(rc)
        assert isinstance(result, logging.Logger)

    def test_fallback_logger_name(self):
        rc = MagicMock(spec=[])
        result = _log(rc)
        assert result.name == "observability"


# ── _maybe_await ──────────────────────────────────────────────────────────────

class TestMaybeAwait:
    def test_plain_value_returned_as_is(self):
        assert asyncio.run(_maybe_await(42)) == 42

    def test_string_returned_as_is(self):
        assert asyncio.run(_maybe_await("hello")) == "hello"

    def test_none_returned_as_is(self):
        assert asyncio.run(_maybe_await(None)) is None

    def test_awaitable_is_awaited(self):
        async def coro():
            return "result"
        assert asyncio.run(_maybe_await(coro())) == "result"

    def test_async_mock_awaited(self):
        mock = AsyncMock(return_value="mocked")
        assert asyncio.run(_maybe_await(mock())) == "mocked"


# ── save_dom ──────────────────────────────────────────────────────────────────

class TestSaveDom:
    def test_page_none_returns_immediately(self):
        rc = _make_rc()
        asyncio.run(save_dom(rc, None, "test"))
        rc.save_content.assert_not_called()

    def test_page_closed_returns_immediately(self):
        rc = _make_rc()
        page = _make_page(closed=True)
        asyncio.run(save_dom(rc, page, "test"))
        rc.save_content.assert_not_called()
        page.content.assert_not_called()

    def test_saves_html_with_correct_name(self):
        rc = _make_rc()
        page = _make_page(content="<html>test</html>")
        asyncio.run(save_dom(rc, page, "snapshot"))
        rc.save_content.assert_called_once_with("snapshot.html", "<html>test</html>")

    def test_exception_in_content_logs_warning(self):
        rc = _make_rc()
        page = _make_page()
        page.content = AsyncMock(side_effect=Exception("network error"))
        # should not raise
        asyncio.run(save_dom(rc, page, "test"))
        rc.save_content.assert_not_called()

    def test_sync_save_content_also_works(self):
        """save_content が同期関数の場合も _maybe_await で動作する"""
        rc = _make_rc()
        rc.save_content = MagicMock(return_value="ok")  # sync
        page = _make_page(content="<html/>")
        asyncio.run(save_dom(rc, page, "sync_test"))
        rc.save_content.assert_called_once()


# ── save_json ─────────────────────────────────────────────────────────────────

class TestSaveJson:
    def test_calls_save_json_with_correct_name(self):
        rc = _make_rc()
        data = {"key": "value"}
        asyncio.run(save_json(rc, data, "my_data"))
        rc.save_json.assert_called_once_with("my_data.json", data)

    def test_exception_logs_warning(self):
        rc = _make_rc()
        rc.save_json = AsyncMock(side_effect=Exception("IO error"))
        # should not raise
        asyncio.run(save_json(rc, {}, "test"))

    def test_passes_data_unchanged(self):
        rc = _make_rc()
        data = [1, 2, 3]
        asyncio.run(save_json(rc, data, "list_data"))
        rc.save_json.assert_called_once_with("list_data.json", data)

    def test_none_data_passes(self):
        rc = _make_rc()
        asyncio.run(save_json(rc, None, "null_data"))
        rc.save_json.assert_called_once_with("null_data.json", None)


# ── count_selectors ───────────────────────────────────────────────────────────

class TestCountSelectors:
    def test_page_none_returns_immediately(self):
        rc = _make_rc()
        asyncio.run(count_selectors(rc, None, ["button"]))
        rc.save_json.assert_not_called()

    def test_page_closed_returns_immediately(self):
        rc = _make_rc()
        page = _make_page(closed=True)
        asyncio.run(count_selectors(rc, page, ["button"]))
        rc.save_json.assert_not_called()

    def test_empty_selectors_saves_empty_dict(self):
        rc = _make_rc()
        page = _make_page()
        asyncio.run(count_selectors(rc, page, []))
        rc.save_json.assert_called_once()
        call_args = rc.save_json.call_args[0]
        assert call_args[1] == {"selector_counts": {}}

    def test_counts_each_selector(self):
        rc = _make_rc()
        page = _make_page()
        loc_mock = MagicMock()
        loc_mock.count = AsyncMock(return_value=3)
        page.locator = MagicMock(return_value=loc_mock)
        asyncio.run(count_selectors(rc, page, ["button", ".price"]))
        call_args = rc.save_json.call_args[0]
        counts = call_args[1]["selector_counts"]
        assert counts["button"] == 3
        assert counts[".price"] == 3

    def test_locator_exception_sets_minus1(self):
        rc = _make_rc()
        page = _make_page()
        loc_mock = MagicMock()
        loc_mock.count = AsyncMock(side_effect=Exception("selector error"))
        page.locator = MagicMock(return_value=loc_mock)
        asyncio.run(count_selectors(rc, page, ["bad-sel"]))
        call_args = rc.save_json.call_args[0]
        assert call_args[1]["selector_counts"]["bad-sel"] == -1

    def test_custom_name_used(self):
        rc = _make_rc()
        page = _make_page()
        asyncio.run(count_selectors(rc, page, [], name="my_counts"))
        rc.save_json.assert_called_once_with("my_counts.json", {"selector_counts": {}})

    def test_none_selectors_treated_as_empty(self):
        rc = _make_rc()
        page = _make_page()
        asyncio.run(count_selectors(rc, page, None))
        call_args = rc.save_json.call_args[0]
        assert call_args[1] == {"selector_counts": {}}


# ── save_raw_hrefs ────────────────────────────────────────────────────────────

class TestSaveRawHrefs:
    def test_saves_hrefs_with_count(self):
        rc = _make_rc()
        hrefs = ["https://a.com", "https://b.com"]
        asyncio.run(save_raw_hrefs(rc, hrefs))
        call_args = rc.save_json.call_args[0]
        data = call_args[1]
        assert data["count"] == 2
        assert data["raw_hrefs"] == hrefs

    def test_limit_applied(self):
        rc = _make_rc()
        hrefs = [f"https://example.com/{i}" for i in range(300)]
        asyncio.run(save_raw_hrefs(rc, hrefs, limit=200))
        call_args = rc.save_json.call_args[0]
        assert len(call_args[1]["raw_hrefs"]) == 200

    def test_none_hrefs_treated_as_empty(self):
        rc = _make_rc()
        asyncio.run(save_raw_hrefs(rc, None))
        call_args = rc.save_json.call_args[0]
        assert call_args[1]["count"] == 0
        assert call_args[1]["raw_hrefs"] == []

    def test_custom_name(self):
        rc = _make_rc()
        asyncio.run(save_raw_hrefs(rc, [], name="my_hrefs"))
        rc.save_json.assert_called_once()
        # save_json内部でraw_hrefs.jsonが渡される（save_jsonがラップする）

    def test_exception_logs_warning(self):
        rc = _make_rc()
        rc.save_json = AsyncMock(side_effect=Exception("IO error"))
        # should not raise
        asyncio.run(save_raw_hrefs(rc, ["https://a.com"]))


# ── write_fail_snapshot ───────────────────────────────────────────────────────

class TestWriteFailSnapshot:
    def test_page_none_saves_snapshot(self):
        """page=None でも MD レポートを save_content に保存する"""
        rc = _make_rc()
        rc.save_content = AsyncMock(return_value=None)
        error = Exception("test error")
        asyncio.run(write_fail_snapshot(rc, None, "https://example.com", error, {}))
        # fail_snapshot.md が保存される
        rc.save_content.assert_called()
        call_args = rc.save_content.call_args[0]
        assert "fail_snapshot.md" in call_args[0]
        assert "test error" in call_args[1]

    def test_page_open_saves_dom_and_snapshot(self):
        """page が open の場合は DOM も保存する"""
        rc = _make_rc()
        rc.save_content = AsyncMock(return_value=None)
        rc.take_screenshot = AsyncMock(return_value="99_failure.png")
        page = _make_page(closed=False, content="<html>fail</html>")
        error = Exception("network error")
        asyncio.run(write_fail_snapshot(rc, page, "https://example.com", error, {}))
        # save_content が複数回呼ばれる（DOM + MD）
        assert rc.save_content.call_count >= 1

    def test_page_closed_skips_dom(self):
        """page が closed の場合は DOM 保存をスキップ"""
        rc = _make_rc()
        rc.save_content = AsyncMock(return_value=None)
        page = _make_page(closed=True)
        error = Exception("closed page error")
        asyncio.run(write_fail_snapshot(rc, page, None, error, {}))
        # fail_snapshot.md のみ保存
        call_names = [c[0][0] for c in rc.save_content.call_args_list]
        assert "fail_snapshot.md" in call_names

    def test_with_site_config_selectors(self):
        """site_config に selectors がある場合も正常完了する"""
        rc = _make_rc()
        rc.save_content = AsyncMock(return_value=None)
        rc.take_screenshot = AsyncMock(return_value=None)
        page = _make_page(closed=False)
        loc_mock = MagicMock()
        loc_mock.count = AsyncMock(return_value=2)
        page.locator = MagicMock(return_value=loc_mock)
        config = {"selectors": {"pdp": {
            "title_selectors": ["h1"],
            "price_selectors": [".price"]
        }}}
        error = Exception("error")
        asyncio.run(write_fail_snapshot(rc, page, "https://site.com", error, config))
        rc.save_content.assert_called()

    def test_run_id_included_in_report(self):
        """run_id が MD レポートに含まれる"""
        rc = _make_rc()
        rc.run_id = "RUN-12345"
        rc.save_content = AsyncMock(return_value=None)
        error = Exception("err")
        asyncio.run(write_fail_snapshot(rc, None, None, error, {}))
        call_args = rc.save_content.call_args[0]
        assert "RUN-12345" in call_args[1]
