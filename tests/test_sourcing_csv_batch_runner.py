"""sourcing_csv_batch_runner のテスト。"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from app.utils.sourcing_csv_batch_runner import (
    process_csv_batch,
    check_csv_header,
    _process_single_row,
)


def test_batch_mixed_complete_partial_invalid():
    """テストケース1: complete/partial/invalid混在3行CSVでjsonlが3行生成される"""
    with TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "test.csv"
        output_file = Path(tmpdir) / "output.jsonl"
        
        # CSVファイルを作成（complete, partial, invalid の3行）
        with csv_file.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["purchase_price", "selling_price", "shipping_cost", "customs_duty", "procurement_fee", "transaction_fee"],
            )
            writer.writeheader()
            # 行0: complete
            writer.writerow({
                "purchase_price": "10000",
                "selling_price": "20000",
                "shipping_cost": "1000",
                "customs_duty": "500",
                "procurement_fee": "0",
                "transaction_fee": "0",
            })
            # 行1: partial（unknown含む）
            writer.writerow({
                "purchase_price": "10000",
                "selling_price": "20000",
                "shipping_cost": "unknown",
                "customs_duty": "500",
                "procurement_fee": "0",
                "transaction_fee": "0",
            })
            # 行2: invalid（purchase_priceが非数値）
            writer.writerow({
                "purchase_price": "abc",
                "selling_price": "20000",
                "shipping_cost": "1000",
                "customs_duty": "500",
                "procurement_fee": "0",
                "transaction_fee": "0",
            })
        
        # バッチ処理実行
        exit_code = process_csv_batch(
            csv_path=csv_file,
            output_path=output_file,
            start_row=0,
            max_rows=None,
        )
        
        assert exit_code == 0
        assert output_file.exists()
        
        # JSON Linesを読み込んで検証
        lines = output_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3
        
        # 各行をパース
        results = [json.loads(line) for line in lines]
        
        # 行0: complete
        assert results[0]["row_index"] == 0
        assert results[0]["profitability"]["status"] == "complete"
        assert results[0]["profitability"]["profit"] is not None
        assert results[0]["profitability"]["profit_rate"] is not None
        
        # 行1: partial
        assert results[1]["row_index"] == 1
        assert results[1]["profitability"]["status"] == "partial"
        assert results[1]["profitability"]["profit"] is None
        assert results[1]["profitability"]["profit_rate"] is None
        assert "missing_fields" in results[1]["profitability"]
        assert results[1]["tier_judgement"]["tier"] == "C"
        
        # 行2: invalid
        assert results[2]["row_index"] == 2
        assert results[2]["profitability"]["status"] == "invalid"
        assert results[2]["tier_judgement"]["tier"] == "D"


def test_batch_header_missing_required_column():
    """テストケース2: ヘッダ不備（purchase_price or selling_price 欠損）で全体失敗する"""
    with TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "test.csv"
        output_file = Path(tmpdir) / "output.jsonl"
        
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
        
        # ヘッダーチェック
        header_ok, header_errors = check_csv_header(csv_file)
        assert header_ok is False
        assert len(header_errors) > 0
        assert any("purchase_price" in error for error in header_errors)
        
        # バッチ処理実行（全体失敗）
        exit_code = process_csv_batch(
            csv_path=csv_file,
            output_path=output_file,
            start_row=0,
            max_rows=None,
        )
        
        assert exit_code == 2
        # 出力ファイルは生成されない（または空）


def test_batch_start_row_and_max_rows():
    """テストケース3: --start-row と --max-rows が指定どおりに効く"""
    with TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "test.csv"
        output_file = Path(tmpdir) / "output.jsonl"
        
        # CSVファイルを作成（5行）
        with csv_file.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["purchase_price", "selling_price", "shipping_cost", "customs_duty", "procurement_fee", "transaction_fee"],
            )
            writer.writeheader()
            for i in range(5):
                writer.writerow({
                    "purchase_price": str(10000 + i * 1000),
                    "selling_price": "20000",
                    "shipping_cost": "1000",
                    "customs_duty": "500",
                    "procurement_fee": "0",
                    "transaction_fee": "0",
                })
        
        # start_row=1, max_rows=2 で実行（行1と行2のみ処理）
        exit_code = process_csv_batch(
            csv_path=csv_file,
            output_path=output_file,
            start_row=1,
            max_rows=2,
        )
        
        assert exit_code == 0
        assert output_file.exists()
        
        # JSON Linesを読み込んで検証
        lines = output_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        
        # 各行をパース
        results = [json.loads(line) for line in lines]
        
        # 行インデックスが1と2であることを確認
        assert results[0]["row_index"] == 1
        assert results[1]["row_index"] == 2
        
        # purchase_priceの値も確認
        assert results[0]["input"]["purchase_price"] == 11000.0
        assert results[1]["input"]["purchase_price"] == 12000.0
