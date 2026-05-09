"""sourcing パイプラインのメインエントリーポイント。

入力JSONを読み込み、利益計算とTier判定を行い、結果をJSONファイルに保存する。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.utils.sourcing_input_schema import validate_sourcing_input
from app.utils.sourcing_profitability import calculate_profitability
from app.utils.sourcing_tier import judge_tier


def run_pipeline(
    input_path: str | Path,
    profitability_output_path: str | Path,
    tier_output_path: str | Path,
) -> None:
    """
    パイプラインを実行する。

    Args:
        input_path: 入力JSONファイルパス
        profitability_output_path: 利益計算結果の出力パス
        tier_output_path: Tier判定結果の出力パス
    """
    input_file = Path(input_path)
    profitability_file = Path(profitability_output_path)
    tier_file = Path(tier_output_path)

    # 出力ディレクトリを作成
    profitability_file.parent.mkdir(parents=True, exist_ok=True)
    tier_file.parent.mkdir(parents=True, exist_ok=True)

    # 入力JSONを読み込み
    try:
        input_data: dict[str, Any] = json.loads(input_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        # ファイルが存在しない場合は invalid として処理
        profitability_result = {
            "status": "invalid",
            "revenue": None,
            "total_cost": None,
            "profit": None,
            "profit_rate": None,
            "errors": [f"入力ファイルが見つかりません: {input_file}"],
        }
        tier_result = {
            "tier": "D",
            "reason": "入力ファイルが見つかりません",
        }
    except json.JSONDecodeError as e:
        # JSON解析エラーの場合も invalid
        profitability_result = {
            "status": "invalid",
            "revenue": None,
            "total_cost": None,
            "profit": None,
            "profit_rate": None,
            "errors": [f"JSON解析エラー: {e}"],
        }
        tier_result = {
            "tier": "D",
            "reason": f"JSON解析エラー: {e}",
        }
    else:
        # 入力検証
        validation_result = validate_sourcing_input(input_data)

        # 利益計算
        profitability_result = calculate_profitability(validation_result.get("normalized"))

        # Tier判定
        tier_result = judge_tier(profitability_result)

    # 結果をJSONファイルに保存
    profitability_file.write_text(
        json.dumps(profitability_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    tier_file.write_text(
        json.dumps(tier_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    """コマンドラインエントリーポイント。"""
    project_root = Path(__file__).parent.parent.parent
    input_path = project_root / "data" / "exports" / "sourcing_input.json"
    profitability_output_path = project_root / "data" / "exports" / "profitability.json"
    tier_output_path = project_root / "data" / "exports" / "tier_judgement.json"

    run_pipeline(
        input_path=input_path,
        profitability_output_path=profitability_output_path,
        tier_output_path=tier_output_path,
    )


if __name__ == "__main__":
    main()
