"""CSVからSourcingInput v0を生成し、sourcing_pipelineを実行するCLI。

標準ライブラリのみを使用。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.utils.sourcing_csv_adapter import parse_csv_row_to_sourcing_input
from app.utils.sourcing_pipeline import run_pipeline


def main() -> None:
    """コマンドラインエントリーポイント。"""
    parser = argparse.ArgumentParser(description="CSVからSourcingInput v0を生成し、利益計算とTier判定を実行する")
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="入力CSVファイルパス",
    )
    parser.add_argument(
        "--row",
        type=int,
        default=0,
        help="処理する行インデックス（0始まり、ヘッダー行は除く、デフォルト: 0）",
    )

    args = parser.parse_args()

    # CSVからSourcingInput v0を生成
    csv_path = Path(args.csv)
    result = parse_csv_row_to_sourcing_input(csv_path, row_index=args.row)

    if result["status"] != "valid":
        print("エラー: CSV解析に失敗しました")
        for error in result["errors"]:
            print(f"  - {error}")
        return

    sourcing_input = result["data"]
    if sourcing_input is None:
        print("エラー: データが生成されませんでした")
        return

    # プロジェクトルートを取得
    project_root = Path(__file__).parent.parent.parent

    # sourcing_input.json を data/exports/ に保存
    exports_dir = project_root / "data" / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    sourcing_input_path = exports_dir / "sourcing_input.json"
    sourcing_input_path.write_text(
        json.dumps(sourcing_input, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"sourcing_input.json を保存しました: {sourcing_input_path}")

    # 既存の sourcing_pipeline を呼び出す
    profitability_output_path = exports_dir / "profitability.json"
    tier_output_path = exports_dir / "tier_judgement.json"

    run_pipeline(
        input_path=sourcing_input_path,
        profitability_output_path=profitability_output_path,
        tier_output_path=tier_output_path,
    )

    print(f"profitability.json を保存しました: {profitability_output_path}")
    print(f"tier_judgement.json を保存しました: {tier_output_path}")


if __name__ == "__main__":
    main()
