"""sourcing パイプラインのテスト。"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from app.utils.sourcing_input_schema import validate_sourcing_input
from app.utils.sourcing_pipeline import run_pipeline
from app.utils.sourcing_profitability import calculate_profitability
from app.utils.sourcing_tier import judge_tier


def test_tier_a_complete():
    """Tier A: complete で A（利益率30%以上）"""
    # 利益率30%以上になるように設定
    # 仕入れ: 10000, 販売: 20000 → 利益率約30%（手数料14.2%考慮）
    # より正確に: 販売20000, 手数料2840, 利益 = 20000 - 10000 - 2840 = 7160
    # 利益率 = 7160 / 20000 = 0.358 = 35.8%
    input_data = {
        "purchase_price": 10000,
        "selling_price": 20000,
        "shipping_cost": 0,
        "customs_duty": 0,
        "procurement_fee": 0,
        "transaction_fee": 0,
    }

    validation = validate_sourcing_input(input_data)
    assert validation["valid"] is True
    assert validation["status"] == "complete"

    profitability = calculate_profitability(validation["normalized"])
    assert profitability["status"] == "complete"
    assert profitability["profit_rate"] is not None
    assert profitability["profit_rate"] >= 0.30

    tier = judge_tier(profitability)
    assert tier["tier"] == "A"


def test_tier_b_complete():
    """Tier B: complete で B（利益率20%以上30%未満）"""
    # 利益率20-30%になるように設定
    # 仕入れ: 15000, 販売: 20000 → 利益率約20%（手数料14.2%考慮）
    # 販売20000, 手数料2840, 利益 = 20000 - 15000 - 2840 = 2160
    # 利益率 = 2160 / 20000 = 0.108 = 10.8% → 低すぎる
    # 仕入れ: 12000, 販売: 20000 → 利益 = 20000 - 12000 - 2840 = 5160
    # 利益率 = 5160 / 20000 = 0.258 = 25.8% → OK
    input_data = {
        "purchase_price": 12000,
        "selling_price": 20000,
        "shipping_cost": 0,
        "customs_duty": 0,
        "procurement_fee": 0,
        "transaction_fee": 0,
    }

    validation = validate_sourcing_input(input_data)
    assert validation["valid"] is True
    assert validation["status"] == "complete"

    profitability = calculate_profitability(validation["normalized"])
    assert profitability["status"] == "complete"
    assert profitability["profit_rate"] is not None
    assert 0.20 <= profitability["profit_rate"] < 0.30

    tier = judge_tier(profitability)
    assert tier["tier"] == "B"


def test_tier_c_partial():
    """Tier C: partial で C（unknown含む）"""
    # unknown を含む入力
    input_data = {
        "purchase_price": 18000,
        "selling_price": 20000,
        "shipping_cost": "unknown",
        "customs_duty": 0,
        "procurement_fee": 0,
        "transaction_fee": 0,
    }

    validation = validate_sourcing_input(input_data)
    assert validation["valid"] is True
    assert validation["status"] == "partial"

    profitability = calculate_profitability(validation["normalized"])
    assert profitability["status"] == "partial"

    # partial の場合、profit と profit_rate は null
    assert profitability["profit"] is None
    assert profitability["profit_rate"] is None

    # revenue と known_cost_sum は確定
    assert profitability["revenue"] is not None
    assert profitability["known_cost_sum"] is not None

    # missing_fields と profit_upper_bound が存在
    assert "missing_fields" in profitability
    assert profitability["missing_fields"] == ["shipping_cost"]
    assert "profit_upper_bound" in profitability
    assert profitability["profit_upper_bound"] is not None

    tier = judge_tier(profitability)
    # partial は原則 Tier C（A/B禁止）
    assert tier["tier"] == "C"


def test_tier_d_invalid():
    """Tier D: invalid で D（必須欠損/不正値）"""
    # 必須項目欠損
    input_data_missing = {
        "selling_price": 20000,
        # purchase_price が欠損
    }

    validation = validate_sourcing_input(input_data_missing)
    assert validation["valid"] is False
    assert validation["status"] == "invalid"

    profitability = calculate_profitability(validation["normalized"])
    assert profitability["status"] == "invalid"

    tier = judge_tier(profitability)
    assert tier["tier"] == "D"

    # 不正値（負の値）
    input_data_negative = {
        "purchase_price": -1000,
        "selling_price": 20000,
    }

    validation_neg = validate_sourcing_input(input_data_negative)
    assert validation_neg["valid"] is False
    assert validation_neg["status"] == "invalid"

    profitability_neg = calculate_profitability(validation_neg["normalized"])
    tier_neg = judge_tier(profitability_neg)
    assert tier_neg["tier"] == "D"


def test_pipeline_integration():
    """パイプライン統合テスト（ファイルI/O含む）"""
    with TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "sourcing_input.json"
        profitability_file = Path(tmpdir) / "profitability.json"
        tier_file = Path(tmpdir) / "tier_judgement.json"

        # テスト用入力データ
        input_data = {
            "purchase_price": 10000,
            "selling_price": 20000,
            "shipping_cost": 0,
            "customs_duty": 0,
            "procurement_fee": 0,
            "transaction_fee": 0,
        }

        input_file.write_text(
            json.dumps(input_data, ensure_ascii=False),
            encoding="utf-8",
        )

        # パイプライン実行
        run_pipeline(
            input_path=input_file,
            profitability_output_path=profitability_file,
            tier_output_path=tier_file,
        )

        # 結果確認
        assert profitability_file.exists()
        assert tier_file.exists()

        profitability_result = json.loads(profitability_file.read_text(encoding="utf-8"))
        tier_result = json.loads(tier_file.read_text(encoding="utf-8"))

        assert profitability_result["status"] == "complete"
        # complete の場合、profit と profit_rate が確定
        assert profitability_result["profit"] is not None
        assert profitability_result["profit_rate"] is not None
        # complete の場合、known_cost_sum と missing_fields は None
        assert profitability_result["known_cost_sum"] is None
        assert profitability_result["missing_fields"] is None

        assert "tier" in tier_result
        assert tier_result["tier"] in ["A", "B", "C", "D"]
