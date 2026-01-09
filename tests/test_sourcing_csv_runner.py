"""sourcing_csv_runner の統合テスト。"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from app.utils.sourcing_csv_runner import main
from app.utils.sourcing_csv_adapter import parse_csv_row_to_sourcing_input
from app.utils.sourcing_pipeline import run_pipeline


def test_csv_runner_complete():
    """統合テスト: completeケース"""
    with TemporaryDirectory() as tmpdir:
        # CSVファイルを作成
        csv_file = Path(tmpdir) / "test.csv"
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
        
        # 一時ディレクトリをプロジェクトルートとして扱う
        # （実際のmain()はプロジェクトルートを取得するが、テストでは一時ディレクトリを使用）
        exports_dir = Path(tmpdir) / "data" / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        
        # CSVからSourcingInput v0を生成
        result = parse_csv_row_to_sourcing_input(csv_file, row_index=0)
        assert result["status"] == "valid"
        assert result["data"] is not None
        
        # sourcing_input.json を保存
        sourcing_input_path = exports_dir / "sourcing_input.json"
        sourcing_input_path.write_text(
            json.dumps(result["data"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        
        # パイプライン実行
        profitability_output_path = exports_dir / "profitability.json"
        tier_output_path = exports_dir / "tier_judgement.json"
        
        run_pipeline(
            input_path=sourcing_input_path,
            profitability_output_path=profitability_output_path,
            tier_output_path=tier_output_path,
        )
        
        # 結果確認
        assert profitability_output_path.exists()
        assert tier_output_path.exists()
        
        profitability_result = json.loads(profitability_output_path.read_text(encoding="utf-8"))
        tier_result = json.loads(tier_output_path.read_text(encoding="utf-8"))
        
        assert profitability_result["status"] == "complete"
        assert profitability_result["profit"] is not None
        assert profitability_result["profit_rate"] is not None
        assert "tier" in tier_result
        assert tier_result["tier"] in ["A", "B", "C", "D"]


def test_csv_runner_partial():
    """統合テスト: partialケース（unknown含む）"""
    with TemporaryDirectory() as tmpdir:
        # CSVファイルを作成（unknown含む）
        csv_file = Path(tmpdir) / "test.csv"
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
        
        # 一時ディレクトリをプロジェクトルートとして扱う
        exports_dir = Path(tmpdir) / "data" / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        
        # CSVからSourcingInput v0を生成
        result = parse_csv_row_to_sourcing_input(csv_file, row_index=0)
        assert result["status"] == "valid"
        assert result["data"] is not None
        
        # sourcing_input.json を保存
        sourcing_input_path = exports_dir / "sourcing_input.json"
        sourcing_input_path.write_text(
            json.dumps(result["data"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        
        # パイプライン実行
        profitability_output_path = exports_dir / "profitability.json"
        tier_output_path = exports_dir / "tier_judgement.json"
        
        run_pipeline(
            input_path=sourcing_input_path,
            profitability_output_path=profitability_output_path,
            tier_output_path=tier_output_path,
        )
        
        # 結果確認
        assert profitability_output_path.exists()
        assert tier_output_path.exists()
        
        profitability_result = json.loads(profitability_output_path.read_text(encoding="utf-8"))
        tier_result = json.loads(tier_output_path.read_text(encoding="utf-8"))
        
        assert profitability_result["status"] == "partial"
        assert profitability_result["profit"] is None
        assert profitability_result["profit_rate"] is None
        assert profitability_result["revenue"] is not None
        assert profitability_result["known_cost_sum"] is not None
        assert profitability_result["missing_fields"] is not None
        assert profitability_result["profit_upper_bound"] is not None
        assert tier_result["tier"] == "C"
