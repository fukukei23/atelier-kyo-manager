"""CSVの全行を評価し、各行のprofitability/tierをJSON Lines（jsonl）に出力するCLI。

標準ライブラリのみを使用。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict

from app.utils.sourcing_csv_adapter import parse_csv_row_dict
from app.utils.sourcing_input_schema import validate_sourcing_input
from app.utils.sourcing_profitability import calculate_profitability
from app.utils.sourcing_tier import judge_tier


def check_csv_header(csv_path: Path) -> tuple[bool, list[str]]:
    """
    CSVヘッダーをチェックし、必須列の存在を確認する。
    
    Args:
        csv_path: CSVファイルパス
        
    Returns:
        (成功フラグ, エラーメッセージリスト)
    """
    errors: list[str] = []
    
    if not csv_path.exists():
        return False, [f"CSVファイルが見つかりません: {csv_path}"]
    
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            
            if fieldnames is None:
                return False, ["CSVヘッダーが読み込めませんでした"]
            
            # 必須列チェック
            required_columns = ["purchase_price", "selling_price"]
            for col in required_columns:
                if col not in fieldnames:
                    errors.append(f"必須列 '{col}' が見つかりません")
            
            if errors:
                return False, errors
            
            return True, []
            
    except Exception as e:
        return False, [f"CSV読み込みエラー: {e}"]


def process_csv_batch(
    csv_path: Path,
    output_path: Path,
    start_row: int = 0,
    max_rows: int | None = None,
) -> int:
    """
    CSVの全行を評価し、JSON Lines形式で出力する。
    
    Args:
        csv_path: 入力CSVファイルパス
        output_path: 出力JSON Linesファイルパス
        start_row: データ行の開始インデックス（デフォルト: 0）
        max_rows: 評価する最大行数（Noneの場合は全行）
        
    Returns:
        exit code（0: 成功, 2: 全体失敗）
    """
    # ヘッダーチェック（Fail-Fast）
    header_ok, header_errors = check_csv_header(csv_path)
    if not header_ok:
        print("エラー: CSVヘッダーに問題があります", file=sys.stderr)
        for error in header_errors:
            print(f"  - {error}", file=sys.stderr)
        return 2
    
    # 出力ディレクトリを作成
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # CSVを読み込んで各行を処理（streaming前提：全行読み込みしない）
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            
            processed_count = 0
            row_index = 0
            target_rows_processed = 0
            
            with output_path.open("w", encoding="utf-8") as out_f:
                # iteratorで1行ずつ処理（メモリに全件保持しない）
                for row in reader:
                    # start_row未満の行はスキップ
                    if row_index < start_row:
                        row_index += 1
                        continue
                    
                    # max_rowsが指定されている場合、上限に達したら終了
                    if max_rows is not None and target_rows_processed >= max_rows:
                        break
                    
                    # 各行を処理（Fail-Soft: エラーでも継続）
                    result = _process_single_row(row_index, row)
                    
                    # JSON Lines形式で出力
                    json_line = json.dumps(result, ensure_ascii=False)
                    out_f.write(json_line + "\n")
                    
                    row_index += 1
                    target_rows_processed += 1
                    processed_count += 1
            
            print(f"処理完了: {processed_count} 行を評価しました", file=sys.stderr)
            print(f"出力先: {output_path}", file=sys.stderr)
            return 0
            
    except Exception as e:
        print(f"エラー: CSV処理中に例外が発生しました: {e}", file=sys.stderr)
        return 2


def _process_single_row(row_index: int, row: Dict[str, Any]) -> Dict[str, Any]:
    """
    単一行を処理し、結果を返す。
    
    重要：invalid行は絶対にdownstream（profitability/tier）に流さない。
    
    Args:
        row_index: 行インデックス（0始まり）
        row: CSV行データ（DictReaderの結果）
        
    Returns:
        {
            "row_index": int,
            "input": dict | None,
            "profitability": dict,
            "tier_judgement": dict
        }
    """
    # CSV行をSourcingInput v0に変換（公開APIを使用）
    adapter_result = parse_csv_row_dict(row)
    
    # invalid行はdownstreamに流さず、ここで処理を終了
    if adapter_result["status"] == "invalid":
        return {
            "row_index": row_index,
            "input": None,
            "profitability": {
                "status": "invalid",
                "errors": adapter_result["errors"],
            },
            "tier_judgement": {
                "tier": "D",
                "reason": f"invalid input: {', '.join(adapter_result['errors'])}",
            },
        }
    
    # valid行のみ処理を続行
    sourcing_input = adapter_result["data"]
    if sourcing_input is None:
        # 念のため、dataがNoneの場合もinvalidとして扱う
        return {
            "row_index": row_index,
            "input": None,
            "profitability": {
                "status": "invalid",
                "errors": ["sourcing_input data is None"],
            },
            "tier_judgement": {
                "tier": "D",
                "reason": "invalid input: sourcing_input data is None",
            },
        }
    
    # 入力検証
    validation_result = validate_sourcing_input(sourcing_input)
    
    # 検証がinvalidの場合もdownstreamに流さない
    if not validation_result["valid"] or validation_result["status"] == "invalid":
        return {
            "row_index": row_index,
            "input": sourcing_input,
            "profitability": {
                "status": "invalid",
                "errors": validation_result["errors"],
            },
            "tier_judgement": {
                "tier": "D",
                "reason": f"invalid input: {', '.join(validation_result['errors'])}",
            },
        }
    
    # 正常行のみ利益計算とTier判定を実行（invalidは絶対に流さない）
    profitability_result = calculate_profitability(validation_result.get("normalized"))
    tier_result = judge_tier(profitability_result)
    
    return {
        "row_index": row_index,
        "input": sourcing_input,
        "profitability": profitability_result,
        "tier_judgement": tier_result,
    }


def main() -> None:
    """コマンドラインエントリーポイント。"""
    parser = argparse.ArgumentParser(
        description="CSVの全行を評価し、各行のprofitability/tierをJSON Lines形式で出力する"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="入力CSVファイルパス",
    )
    parser.add_argument(
        "--out-jsonl",
        type=str,
        default=None,
        help="出力JSON Linesファイルパス（デフォルト: data/exports/sourcing_results.jsonl）",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=0,
        help="データ行の開始インデックス（0始まり、デフォルト: 0）",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="評価する最大行数（指定しない場合は全行）",
    )
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    
    # 出力パスの決定
    if args.out_jsonl:
        output_path = Path(args.out_jsonl)
    else:
        project_root = Path(__file__).parent.parent.parent
        output_path = project_root / "data" / "exports" / "sourcing_results.jsonl"
    
    # バッチ処理実行
    exit_code = process_csv_batch(
        csv_path=csv_path,
        output_path=output_path,
        start_row=args.start_row,
        max_rows=args.max_rows,
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
