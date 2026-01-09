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

from app.utils.sourcing_csv_adapter import parse_csv_row_to_sourcing_input
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
    
    # CSVを読み込んで各行を処理
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            # 処理範囲を決定
            end_row = len(rows)
            if max_rows is not None:
                end_row = min(start_row + max_rows, len(rows))
            
            processed_count = 0
            
            with output_path.open("w", encoding="utf-8") as out_f:
                for row_index in range(start_row, end_row):
                    # 各行を処理（Fail-Soft: エラーでも継続）
                    result = _process_single_row(row_index, rows[row_index])
                    
                    # JSON Lines形式で出力
                    json_line = json.dumps(result, ensure_ascii=False)
                    out_f.write(json_line + "\n")
                    
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
    # CSV行をSourcingInput v0に変換
    # 注意: parse_csv_row_to_sourcing_input はファイルパスと行インデックスを要求するが、
    # ここでは既に読み込んだ行データを使う必要がある
    # そのため、一時的に行データを辞書として扱う
    
    # 行データから直接SourcingInput v0を構築
    sourcing_input: Dict[str, Any] = {}
    errors: list[str] = []
    
    # 必須列チェック
    required_columns = ["purchase_price", "selling_price"]
    for col in required_columns:
        if col not in row:
            errors.append(f"必須列 '{col}' が見つかりません")
    
    if errors:
        return {
            "row_index": row_index,
            "input": None,
            "profitability": {
                "status": "invalid",
                "errors": errors,
            },
            "tier_judgement": {
                "tier": "D",
                "reason": f"CSV行の解析エラー: {', '.join(errors)}",
            },
        }
    
    # 各フィールドをパース（sourcing_csv_adapterのロジックを再利用）
    from app.utils.sourcing_csv_adapter import _normalize_unknown_value, _parse_numeric_value
    
    # purchase_price（必須）
    purchase_price_str = _normalize_unknown_value(row.get("purchase_price"))
    if purchase_price_str == "unknown":
        sourcing_input["purchase_price"] = "unknown"
    else:
        purchase_price, success = _parse_numeric_value(purchase_price_str)
        if not success:
            return {
                "row_index": row_index,
                "input": None,
                "profitability": {
                    "status": "invalid",
                    "errors": [f"purchase_price が数値として解釈できません: {row.get('purchase_price')}"],
                },
                "tier_judgement": {
                    "tier": "D",
                    "reason": "purchase_price が数値として解釈できません",
                },
            }
        sourcing_input["purchase_price"] = purchase_price
    
    # selling_price（必須）
    selling_price_str = _normalize_unknown_value(row.get("selling_price"))
    if selling_price_str == "unknown":
        sourcing_input["selling_price"] = "unknown"
    else:
        selling_price, success = _parse_numeric_value(selling_price_str)
        if not success:
            return {
                "row_index": row_index,
                "input": None,
                "profitability": {
                    "status": "invalid",
                    "errors": [f"selling_price が数値として解釈できません: {row.get('selling_price')}"],
                },
                "tier_judgement": {
                    "tier": "D",
                    "reason": "selling_price が数値として解釈できません",
                },
            }
        sourcing_input["selling_price"] = selling_price
    
    # オプション項目
    optional_fields = ["shipping_cost", "customs_duty", "procurement_fee", "transaction_fee"]
    for field in optional_fields:
        value_str = _normalize_unknown_value(row.get(field))
        if value_str == "unknown":
            sourcing_input[field] = "unknown"
        else:
            value, success = _parse_numeric_value(value_str)
            if success:
                sourcing_input[field] = value
            else:
                sourcing_input[field] = "unknown"
    
    # 入力検証
    validation_result = validate_sourcing_input(sourcing_input)
    
    # 利益計算
    profitability_result = calculate_profitability(validation_result.get("normalized"))
    
    # Tier判定
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
