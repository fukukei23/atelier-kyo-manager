"""CSV行をSourcingInput v0（フラット）に変換するモジュール。

標準ライブラリのcsvモジュールのみを使用。
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple


def _normalize_unknown_value(value: str | None) -> str | None:
    """
    空欄/unknown/-/N/A 等を "unknown" に正規化する。
    
    Args:
        value: CSVのセル値
        
    Returns:
        "unknown" または元の値（数値文字列の場合）
    """
    if value is None:
        return "unknown"
    
    value_str = str(value).strip()
    
    if not value_str:
        return "unknown"
    
    # 未知値のパターンを正規化
    unknown_patterns = ["unknown", "-", "n/a", "na", "n.a.", "null", "none", "?", "??"]
    if value_str.lower() in unknown_patterns:
        return "unknown"
    
    return value_str


def _parse_numeric_value(value: str | None) -> Tuple[float | None, bool]:
    """
    数値文字列をパースする。カンマ付き（"1,000"）も許容。
    
    Args:
        value: 数値文字列
        
    Returns:
        (パース結果, 成功フラグ)
    """
    if value is None:
        return None, False
    
    value_str = str(value).strip()
    
    # unknown の場合は None を返す
    if _normalize_unknown_value(value_str) == "unknown":
        return None, False
    
    # カンマを除去
    value_str = value_str.replace(",", "")
    
    try:
        return float(value_str), True
    except ValueError:
        return None, False


def parse_csv_row_to_sourcing_input(
    csv_path: str | Path,
    row_index: int = 0,
) -> Dict[str, Any]:
    """
    CSVファイルの指定行をSourcingInput v0（フラット）に変換する。
    
    Args:
        csv_path: CSVファイルパス
        row_index: 行インデックス（0始まり、ヘッダー行は除く）
        
    Returns:
        {
            "status": "valid" | "invalid",
            "data": Dict[str, Any] | None,  # SourcingInput v0形式
            "errors": List[str]
        }
    """
    csv_file = Path(csv_path)
    errors: List[str] = []
    
    # ファイル存在チェック
    if not csv_file.exists():
        return {
            "status": "invalid",
            "data": None,
            "errors": [f"CSVファイルが見つかりません: {csv_file}"],
        }
    
    try:
        with csv_file.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            # 行インデックスチェック
            if row_index < 0 or row_index >= len(rows):
                return {
                    "status": "invalid",
                    "data": None,
                    "errors": [f"行インデックス {row_index} が範囲外です（有効範囲: 0-{len(rows)-1}）"],
                }
            
            row = rows[row_index]
            
            # 必須列チェック
            required_columns = ["purchase_price", "selling_price"]
            for col in required_columns:
                if col not in row:
                    errors.append(f"必須列 '{col}' が見つかりません")
            
            if errors:
                return {
                    "status": "invalid",
                    "data": None,
                    "errors": errors,
                }
            
            # 各フィールドをパース
            sourcing_input: Dict[str, Any] = {}
            
            # purchase_price（必須）
            purchase_price_str = _normalize_unknown_value(row.get("purchase_price"))
            if purchase_price_str == "unknown":
                sourcing_input["purchase_price"] = "unknown"
            else:
                purchase_price, success = _parse_numeric_value(purchase_price_str)
                if not success:
                    errors.append(f"purchase_price が数値として解釈できません: {row.get('purchase_price')}")
                    return {
                        "status": "invalid",
                        "data": None,
                        "errors": errors,
                    }
                sourcing_input["purchase_price"] = purchase_price
            
            # selling_price（必須）
            selling_price_str = _normalize_unknown_value(row.get("selling_price"))
            if selling_price_str == "unknown":
                sourcing_input["selling_price"] = "unknown"
            else:
                selling_price, success = _parse_numeric_value(selling_price_str)
                if not success:
                    errors.append(f"selling_price が数値として解釈できません: {row.get('selling_price')}")
                    return {
                        "status": "invalid",
                        "data": None,
                        "errors": errors,
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
                        # オプション項目は unknown として扱う
                        sourcing_input[field] = "unknown"
            
            return {
                "status": "valid",
                "data": sourcing_input,
                "errors": [],
            }
            
    except Exception as e:
        return {
            "status": "invalid",
            "data": None,
            "errors": [f"CSV読み込みエラー: {e}"],
        }
