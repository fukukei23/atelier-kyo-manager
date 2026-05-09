"""入力JSONの検証・正規化を行うモジュール。

Fail-Fast: 不正入力は例外で落とさず、検証結果を返す。
"""

from __future__ import annotations

from typing import Any, Literal


def validate_sourcing_input(data: dict[str, Any]) -> dict[str, Any]:
    """
    入力JSONを検証し、検証結果を返す。

    Args:
        data: 入力JSON辞書

    Returns:
        {
            "valid": bool,
            "status": "complete" | "partial" | "invalid",
            "errors": List[str],
            "normalized": Dict[str, Any] | None
        }
    """
    errors = []
    normalized: dict[str, Any] = {}
    has_unknown = False

    # 必須項目チェック
    required_fields = ["purchase_price", "selling_price"]
    for field in required_fields:
        if field not in data:
            errors.append(f"必須項目 '{field}' が欠損しています")
            return {
                "valid": False,
                "status": "invalid",
                "errors": errors,
                "normalized": None,
            }

    # 型・範囲チェック
    numeric_fields = {
        "purchase_price": (0.0, None),  # 0以上
        "selling_price": (0.0, None),  # 0以上
        "shipping_cost": (0.0, None),  # 0以上（オプション、デフォルト0）
        "customs_duty": (0.0, None),  # 0以上（オプション、デフォルト0）
        "procurement_fee": (0.0, None),  # 0以上（オプション、デフォルト0）
        "transaction_fee": (0.0, None),  # 0以上（オプション、デフォルト0）
    }

    for field, (min_val, max_val) in numeric_fields.items():
        value = data.get(field, 0.0 if field not in required_fields else None)

        if value is None:
            if field in required_fields:
                errors.append(f"必須項目 '{field}' が None です")
                return {
                    "valid": False,
                    "status": "invalid",
                    "errors": errors,
                    "normalized": None,
                }
            normalized[field] = 0.0
            continue

        # "unknown" チェック
        if isinstance(value, str) and value.lower() == "unknown":
            has_unknown = True
            normalized[field] = None  # 計算不可として None を保持
            continue

        # 数値型チェック
        if not isinstance(value, (int, float)):
            errors.append(f"'{field}' は数値または 'unknown' である必要があります（現在: {type(value).__name__}）")
            return {
                "valid": False,
                "status": "invalid",
                "errors": errors,
                "normalized": None,
            }

        # 範囲チェック
        num_value = float(value)
        if min_val is not None and num_value < min_val:
            errors.append(f"'{field}' は {min_val} 以上である必要があります（現在: {num_value}）")
            return {
                "valid": False,
                "status": "invalid",
                "errors": errors,
                "normalized": None,
            }
        if max_val is not None and num_value > max_val:
            errors.append(f"'{field}' は {max_val} 以下である必要があります（現在: {num_value}）")
            return {
                "valid": False,
                "status": "invalid",
                "errors": errors,
                "normalized": None,
            }

        normalized[field] = num_value

    # 必須項目が unknown の場合は invalid
    if normalized.get("purchase_price") is None or normalized.get("selling_price") is None:
        errors.append("必須項目（purchase_price または selling_price）が 'unknown' です")
        return {
            "valid": False,
            "status": "invalid",
            "errors": errors,
            "normalized": None,
        }

    # ステータス決定
    if has_unknown:
        status: Literal["complete", "partial", "invalid"] = "partial"
    else:
        status = "complete"

    return {
        "valid": True,
        "status": status,
        "errors": [],
        "normalized": normalized,
    }
