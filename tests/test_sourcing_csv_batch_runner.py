"""Tests for sourcing_csv_batch_runner module."""
import json
from pathlib import Path

import pytest

from app.utils.sourcing_csv_batch_runner import (
    check_csv_header,
    process_csv_batch,
    _process_single_row,
)


class TestCheckCsvHeader:
    def test_valid_header(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("purchase_price,selling_price,brand\n100,200,Test\n", encoding="utf-8")
        ok, errors = check_csv_header(csv_file)
        assert ok is True
        assert errors == []

    def test_missing_required_column(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,brand\nitem,test\n", encoding="utf-8")
        ok, errors = check_csv_header(csv_file)
        assert ok is False
        assert len(errors) > 0

    def test_file_not_found(self):
        ok, errors = check_csv_header(Path("/nonexistent/file.csv"))
        assert ok is False

    def test_empty_csv(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("", encoding="utf-8")
        ok, errors = check_csv_header(csv_file)
        assert ok is False


class TestProcessSingleRow:
    def test_valid_row(self):
        row = {"purchase_price": "100", "selling_price": "200"}
        result = _process_single_row(0, row)
        assert result["row_index"] == 0
        assert result["input"] is not None
        assert result["profitability"]["status"] in ("valid", "partial")

    def test_invalid_row_missing_columns(self):
        row = {"name": "test"}
        result = _process_single_row(0, row)
        assert result["profitability"]["status"] == "invalid"
        assert result["tier_judgement"]["tier"] == "D"

    def test_invalid_row_bad_prices(self):
        row = {"purchase_price": "abc", "selling_price": "xyz"}
        result = _process_single_row(0, row)
        assert result["profitability"]["status"] == "invalid"


class TestProcessCsvBatch:
    def _make_csv(self, tmp_path, rows):
        csv_file = tmp_path / "batch.csv"
        header = "purchase_price,selling_price"
        lines = [header] + [f"{r[0]},{r[1]}" for r in rows]
        csv_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return csv_file

    def test_batch_processes_all_rows(self, tmp_path):
        csv_file = self._make_csv(tmp_path, [(100, 200), (300, 400)])
        output = tmp_path / "out.jsonl"
        exit_code = process_csv_batch(csv_file, output)
        assert exit_code == 0
        lines = output.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            data = json.loads(line)
            assert "profitability" in data
            assert "tier_judgement" in data

    def test_batch_with_max_rows(self, tmp_path):
        csv_file = self._make_csv(tmp_path, [(100, 200), (300, 400), (500, 600)])
        output = tmp_path / "out.jsonl"
        exit_code = process_csv_batch(csv_file, output, max_rows=2)
        assert exit_code == 0
        lines = output.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

    def test_batch_with_start_row(self, tmp_path):
        csv_file = self._make_csv(tmp_path, [(100, 200), (300, 400), (500, 600)])
        output = tmp_path / "out.jsonl"
        exit_code = process_csv_batch(csv_file, output, start_row=1)
        assert exit_code == 0
        lines = output.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

    def test_bad_header_returns_error(self, tmp_path):
        csv_file = tmp_path / "bad.csv"
        csv_file.write_text("name,brand\ntest,val\n", encoding="utf-8")
        output = tmp_path / "out.jsonl"
        exit_code = process_csv_batch(csv_file, output)
        assert exit_code == 2

    def test_creates_output_directory(self, tmp_path):
        csv_file = self._make_csv(tmp_path, [(100, 200)])
        output = tmp_path / "subdir" / "deep" / "out.jsonl"
        exit_code = process_csv_batch(csv_file, output)
        assert exit_code == 0
        assert output.exists()
