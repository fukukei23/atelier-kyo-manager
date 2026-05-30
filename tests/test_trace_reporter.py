"""Tests for app/utils/trace_reporter.py - helper functions only"""

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.utils.trace_reporter import (
    _event_time_ms,
    _find_all,
    _fmt_epoch_ms,
    _load_jsonl,
    _summarize,
)


def _make_zip(files: dict[str, str]) -> io.BytesIO:
    """インメモリ zipfile を作成"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    buf.seek(0)
    return buf


# ---------- _fmt_epoch_ms ----------


class TestFmtEpochMs:
    def test_formats_valid_timestamp(self):
        result = _fmt_epoch_ms(1735689600000)  # 2025-01-01 00:00:00 UTC
        assert result == "2025-01-01T00:00:00Z"

    def test_returns_dash_for_none(self):
        assert _fmt_epoch_ms(None) == "-"

    def test_returns_dash_for_invalid_int(self):
        assert _fmt_epoch_ms(-1) == "-"

    def test_returns_invalid_for_pre_2001(self):
        result = _fmt_epoch_ms(50000000000)
        assert "Invalid" in result

    def test_handles_float_timestamp(self):
        result = _fmt_epoch_ms(1735689600000.0)
        assert result == "2025-01-01T00:00:00Z"


# ---------- _event_time_ms ----------


class TestEventTimeMs:
    def test_extracts_time_field(self):
        event = {"time": 1735689600000}
        assert _event_time_ms(event) == 1735689600000

    def test_extracts_timestamp_field_usec(self):
        event = {"timestamp": 1735689600000000}  # usec
        result = _event_time_ms(event)
        assert result == 1735689600000  # divided by 1000

    def test_extracts_ts_field_divisible(self):
        # ts is divided by 1000 if it's treated as usec
        event = {"ts": 2000000000}
        result = _event_time_ms(event)
        assert result == 2000000  # divided by 1000

    def test_extracts_start_time_field(self):
        event = {"startTime": 1735689600000}
        assert _event_time_ms(event) == 1735689600000

    def test_returns_minus_one_when_missing(self):
        event = {"type": "action"}
        assert _event_time_ms(event) == -1

    def test_prefers_time_over_timestamp(self):
        event = {"time": 1000, "timestamp": 2000}
        assert _event_time_ms(event) == 1000

    def test_handles_usec_overflow(self):
        # If timestamp looks like usec (> 9466848000000 = year 2270), it's divided
        event = {"timestamp": 1735689600000000}  # usec
        result = _event_time_ms(event)
        assert result == 1735689600000


# ---------- _summarize ----------


class TestSummarize:
    def test_counts_navigation_events(self):
        events = [
            {"type": "navigation", "url": "https://example.com/a"},
            {"type": "navigation", "url": "https://example.com/b"},
            {"type": "NAVIGATION"},  # case insensitive
        ]
        result = _summarize(events)
        assert result["navigations"] == 3
        assert result["event_count"] == 3

    def test_counts_actions_by_api_name(self):
        events = [
            {"type": "action", "apiName": "goto", "params": {"url": "https://ex.com"}},
            {"type": "action", "apiName": "click"},
            {"type": "action", "apiName": "click"},
        ]
        result = _summarize(events)
        assert result["actions"]["goto"] == 1
        assert result["actions"]["click"] == 2

    def test_counts_request_failed(self):
        events = [
            {"type": "request", "error": None},
            {"type": "request", "error": "timeout"},
            {"type": "request"},
        ]
        result = _summarize(events)
        assert result["requests"] == 3
        assert result["request_failed"] == 1

    def test_counts_console_errors(self):
        events = [
            {"type": "console", "params": {"type": "log"}},
            {"type": "console", "params": {"type": "error"}},
            {"type": "console", "params": {"type": "assert"}},
            {"type": "console", "params": {"type": "warn"}},
        ]
        result = _summarize(events)
        assert result["console_errors"] == 2  # error + assert only

    def test_calculates_duration(self):
        events = [
            {"type": "action", "time": 1000000000000, "apiName": "a"},
            {"type": "action", "time": 1000000010000, "apiName": "b"},
        ]
        result = _summarize(events)
        assert result["duration_ms"] == 10000

    def test_extracts_last_url_from_navigation(self):
        events = [
            {"type": "navigation", "url": "https://first.com"},
            {"type": "navigation", "url": "https://last.com"},
        ]
        result = _summarize(events)
        assert "https://last.com" in result["last_url"]

    def test_extracts_last_url_from_action_params(self):
        events = [
            {"type": "action", "apiName": "goto", "params": {"url": "https://first.com"}},
            {"type": "action", "apiName": "goto", "params": {"url": "https://last.com"}},
        ]
        result = _summarize(events)
        assert "https://last.com" in result["last_url"]

    def test_handles_empty_events(self):
        result = _summarize([])
        assert result["event_count"] == 0
        assert result["time_start"] is None
        assert result["time_end"] is None
        assert result["duration_ms"] is None

    def test_wall_time_start_ms_used(self):
        events = [{"type": "action", "apiName": "a"}]
        result = _summarize(events, wall_time_start_ms=1000000000000)
        assert result["actions"]["a"] == 1

    def test_nav_type_alias(self):
        """'nav' は 'navigation' に正規化される"""
        events = [{"type": "nav", "url": "https://nav.com"}]
        result = _summarize(events)
        assert result["navigations"] == 1

    def test_timestamp_usec_field(self):
        """timestamp フィールドが 大きな値でも処理できる"""
        events = [{"timestamp": 9466848000001000}]  # usec
        result = _summarize(events)
        assert result["event_count"] == 1

    def test_ts_usec_field(self):
        """ts フィールドが usec として処理される"""
        events = [{"ts": 5000000000}]
        result = _summarize(events)
        assert result["event_count"] == 1

    def test_metadata_start_time(self):
        """metadata.startTime フィールドが処理される"""
        events = [{"metadata": {"startTime": 1000000000000}}]
        result = _summarize(events)
        assert result["event_count"] == 1

    def test_action_with_error(self):
        """action event に error がある場合カウントされる"""
        events = [{"type": "action", "apiName": "fill", "error": "timeout"}]
        result = _summarize(events)
        assert result["errors"] == 1

    def test_large_ts_usec_conversion(self):
        """ts > 9466848000000 の場合 1000 で割る"""
        large_ts = 9466848000001 * 1000
        events = [{"ts": large_ts}]
        result = _summarize(events)
        assert isinstance(result, dict)


# ── _load_jsonl ───────────────────────────────────────────────────────────────

class TestLoadJsonl:
    def test_missing_file_returns_empty(self):
        buf = _make_zip({})
        with zipfile.ZipFile(buf) as zf:
            result = _load_jsonl(zf, "missing.jsonl")
        assert result == []

    def test_loads_valid_jsonl(self):
        content = '{"a": 1}\n{"b": 2}\n'
        buf = _make_zip({"trace.jsonl": content})
        with zipfile.ZipFile(buf) as zf:
            result = _load_jsonl(zf, "trace.jsonl")
        assert len(result) == 2
        assert result[0] == {"a": 1}

    def test_skips_invalid_json_lines(self):
        content = '{"valid": 1}\nnot json\n{"also": 2}\n'
        buf = _make_zip({"trace.jsonl": content})
        with zipfile.ZipFile(buf) as zf:
            result = _load_jsonl(zf, "trace.jsonl")
        assert len(result) == 2

    def test_skips_empty_lines(self):
        content = '{"a": 1}\n\n\n{"b": 2}\n'
        buf = _make_zip({"trace.jsonl": content})
        with zipfile.ZipFile(buf) as zf:
            result = _load_jsonl(zf, "trace.jsonl")
        assert len(result) == 2

    def test_skips_non_dict_json(self):
        content = '[1,2,3]\n{"dict": true}\n'
        buf = _make_zip({"trace.jsonl": content})
        with zipfile.ZipFile(buf) as zf:
            result = _load_jsonl(zf, "trace.jsonl")
        assert len(result) == 1

    def test_tries_multiple_candidates(self):
        content = '{"found": true}\n'
        buf = _make_zip({"second.jsonl": content})
        with zipfile.ZipFile(buf) as zf:
            result = _load_jsonl(zf, "first.jsonl", "second.jsonl")
        assert result == [{"found": True}]

    def test_returns_first_non_empty(self):
        buf = _make_zip({
            "first.jsonl": '{"x": 1}\n',
            "second.jsonl": '{"y": 2}\n',
        })
        with zipfile.ZipFile(buf) as zf:
            result = _load_jsonl(zf, "first.jsonl", "second.jsonl")
        assert result == [{"x": 1}]


# ── _find_all ─────────────────────────────────────────────────────────────────

class TestFindAll:
    def test_finds_files_by_extension(self):
        buf = _make_zip({"a.trace": "", "b.trace": "", "c.other": ""})
        with zipfile.ZipFile(buf) as zf:
            result = _find_all(zf, ".trace")
        assert "a.trace" in result
        assert "b.trace" in result
        assert "c.other" not in result

    def test_prefix_filter(self):
        buf = _make_zip({"traces/a.trace": "", "other/b.trace": ""})
        with zipfile.ZipFile(buf) as zf:
            result = _find_all(zf, ".trace", prefix="traces/")
        assert "traces/a.trace" in result
        assert "other/b.trace" not in result

    def test_empty_zip_returns_empty(self):
        buf = _make_zip({})
        with zipfile.ZipFile(buf) as zf:
            result = _find_all(zf, ".trace")
        assert result == []

    def test_no_prefix_matches_all(self):
        buf = _make_zip({"dir1/a.png": "", "dir2/b.png": ""})
        with zipfile.ZipFile(buf) as zf:
            result = _find_all(zf, ".png")
        assert len(result) == 2

    def test_no_matching_extension(self):
        buf = _make_zip({"a.txt": "", "b.log": ""})
        with zipfile.ZipFile(buf) as zf:
            result = _find_all(zf, ".trace")
        assert result == []
