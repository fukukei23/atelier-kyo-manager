"""sourcing_csv_adapter のテスト。"""
from __future__ import annotations

import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from app.utils.sourcing_csv_adapter import (
    parse_csv_row_to_sourcing_input,
    _normalize_unknown_value,
    _parse_numeric_value,
)


def test_normalize_unknown_value():
    """unknown値の正規化テスト"""
    assert _normalize_unknown_value(None) == "unknown"
    assert _normalize_unknown_value("") == "unknown"
    assert _normalize_unknown_value("   ") == "unknown"
    assert _normalize_unknown_value("unknown") == "unknown"
    assert _normalize_unknown_value("UNKNOWN") == "unknown"
    assert _normalize_unknown_value("-") == "unknown"
    assert _normalize_unknown_value("N/A") == "unknown"
    assert _normalize_unknown_value("n/a") == "unknown"
    assert _normalize_unknown_value("null") == "unknown"
    assert _normalize_unknown_value("1000") == "1000"
    assert _normalize_unknown_value("1,000") == "1,000"


def test_parse_numeric_value():
    """数値パーステスト"""
    # 正常ケース
    value, success = _parse_numeric_value("1000")
    assert success is True
    assert value == 1000.0
    
    # カンマ付き
    value, success = _parse_numeric_value("1,000")
    assert success is True
    assert value == 1000.0
    
    # 小数点
    value, success = _parse_numeric_value("1234.56")
    assert success is True
    assert value == 1234.56
    
    # unknown
    value, success = _parse_numeric_value("unknown")
    assert success is False
    assert value is None
    
    # 空欄
    value, success = _parse_numeric_value("")
    assert success is False
    assert value is None
    
    # 非数値
    value, success = _parse_numeric_value("abc")
    assert success is False
    assert value is None


def test_parse_csv_row_valid_complete():
    """正常ケース: complete（全項目が数値）"""
    with TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "test.csv"
        
        # CSVファイルを作成
        with csv_file.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["purchase_price", "selling_price", "shipping_cost", "customs_duty", "procurement_fee", "transaction_fee"],
            )
            writer.writeheader()
            writer.writerow({
                "purchase_price": "10000",
                "selling_price": "20000",
                "shipping_cost": "1000",
                "customs_duty": "500",
                "procurement_fee": "0",
                "transaction_fee": "0",
            })
        
        result = parse_csv_row_to_sourcing_input(csv_file, row_index=0)
        
        assert result["status"] == "valid"
        assert result["data"] is not None
        assert result["errors"] == []
        
        data = result["data"]
        assert data["purchase_price"] == 10000.0
        assert data["selling_price"] == 20000.0
        assert data["shipping_cost"] == 1000.0
        assert data["customs_duty"] == 500.0
        assert data["procurement_fee"] == 0.0
        assert data["transaction_fee"] == 0.0


def test_parse_csv_row_valid_partial():
    """正常ケース: partial（unknown含む）"""
    with TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "test.csv"
        
        # CSVファイルを作成
        with csv_file.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["purchase_price", "selling_price", "shipping_cost", "customs_duty", "procurement_fee", "transaction_fee"],
            )
            writer.writeheader()
            writer.writerow({
                "purchase_price": "10000",
                "selling_price": "20000",
                "shipping_cost": "unknown",
                "customs_duty": "-",
                "procurement_fee": "N/A",
                "transaction_fee": "0",
            })
        
        result = parse_csv_row_to_sourcing_input(csv_file, row_index=0)
        
        assert result["status"] == "valid"
        assert result["data"] is not None
        assert result["errors"] == []
        
        data = result["data"]
        assert data["purchase_price"] == 10000.0
        assert data["selling_price"] == 20000.0
        assert data["shipping_cost"] == "unknown"
        assert data["customs_duty"] == "unknown"
        assert data["procurement_fee"] == "unknown"
        assert data["transaction_fee"] == 0.0


def test_parse_csv_row_invalid_missing_column():
    """エラーケース: 必須列が欠損"""
    with TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "test.csv"
        
        # CSVファイルを作成（purchase_priceが欠損）
        with csv_file.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["selling_price", "shipping_cost"],
            )
            writer.writeheader()
            writer.writerow({
                "selling_price": "20000",
                "shipping_cost": "1000",
            })
        
        result = parse_csv_row_to_sourcing_input(csv_file, row_index=0)
        
        assert result["status"] == "invalid"
        assert result["data"] is None
        assert len(result["errors"]) > 0
        assert any("purchase_price" in error for error in result["errors"])


def test_parse_csv_row_invalid_non_numeric():
    """エラーケース: 必須列が非数値"""
    with TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "test.csv"
        
        # CSVファイルを作成（purchase_priceが非数値）
        with csv_file.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["purchase_price", "selling_price"],
            )
            writer.writeheader()
            writer.writerow({
                "purchase_price": "abc",
                "selling_price": "20000",
            })
        
        result = parse_csv_row_to_sourcing_input(csv_file, row_index=0)
        
        assert result["status"] == "invalid"
        assert result["data"] is None
        assert len(result["errors"]) > 0
        assert any("purchase_price" in error and "数値" in error for error in result["errors"])


def test_parse_csv_row_comma_separated_numbers():
    """カンマ区切り数値のパーステスト"""
    with TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "test.csv"
        
        # CSVファイルを作成（カンマ区切り数値）
        with csv_file.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["purchase_price", "selling_price", "shipping_cost"],
            )
            writer.writeheader()
            writer.writerow({
                "purchase_price": "10,000",
                "selling_price": "20,000",
                "shipping_cost": "1,500",
            })
        
        result = parse_csv_row_to_sourcing_input(csv_file, row_index=0)
        
        assert result["status"] == "valid"
        assert result["data"] is not None
        
        data = result["data"]
        assert data["purchase_price"] == 10000.0
        assert data["selling_price"] == 20000.0
        assert data["shipping_cost"] == 1500.0


def test_parse_csv_row_row_index_out_of_range():
    """エラーケース: 行インデックスが範囲外"""
    with TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "test.csv"
        
        # CSVファイルを作成（1行のみ）
        with csv_file.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["purchase_price", "selling_price"],
            )
            writer.writeheader()
            writer.writerow({
                "purchase_price": "10000",
                "selling_price": "20000",
            })
        
        # 範囲外の行インデックス
        result = parse_csv_row_to_sourcing_input(csv_file, row_index=10)
        
        assert result["status"] == "invalid"
        assert result["data"] is None
        assert len(result["errors"]) > 0
        assert any("範囲外" in error for error in result["errors"])
