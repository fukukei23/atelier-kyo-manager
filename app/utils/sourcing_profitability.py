"""利益計算を行うモジュール。

既存の app.core.pricing.calculator を使用して利益を計算する。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from app.core.pricing.calculator import calculate_pricing
from app.core.pricing.schemas import PricingInput


def calculate_profitability(normalized_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    正規化された入力データから利益を計算する。
    
    Args:
        normalized_data: validate_sourcing_input の normalized 結果
        
    Returns:
        {
            "status": "complete" | "partial" | "invalid",
            "revenue": float | None,
            "total_cost": float | None,
            "profit": float | None,
            "profit_rate": float | None,
            "errors": List[str]
        }
    """
    if normalized_data is None:
        return {
            "status": "invalid",
            "revenue": None,
            "total_cost": None,
            "profit": None,
            "profit_rate": None,
            "errors": ["入力データが無効です"],
        }
    
    # unknown が含まれている場合は partial
    has_unknown = any(v is None for v in normalized_data.values())
    
    if has_unknown:
        # partial の場合、計算可能な項目のみで計算を試みる
        # ただし、必須項目（purchase_price, selling_price）が None の場合は計算不可
        if normalized_data.get("purchase_price") is None or normalized_data.get("selling_price") is None:
            return {
                "status": "partial",
                "revenue": None,
                "total_cost": None,
                "profit": None,
                "profit_rate": None,
                "errors": ["必須項目が 'unknown' のため計算できません"],
            }
        
        # 計算可能な項目のみで計算
        # unknown の項目は 0 として扱う（仕様: 自動補正禁止のため、計算結果も partial として扱う）
        inp = PricingInput(
            purchase_price=normalized_data.get("purchase_price", 0.0) or 0.0,
            selling_price=normalized_data.get("selling_price", 0.0) or 0.0,
            shipping_cost=normalized_data.get("shipping_cost", 0.0) or 0.0,
            customs_duty=normalized_data.get("customs_duty", 0.0) or 0.0,
            procurement_fee=normalized_data.get("procurement_fee", 0.0) or 0.0,
            transaction_fee=normalized_data.get("transaction_fee", 0.0) or 0.0,
        )
        
        result = calculate_pricing(inp)
        
        return {
            "status": "partial",
            "revenue": result.revenue,
            "total_cost": result.total_cost,
            "profit": result.profit,
            "profit_rate": result.profit_rate,
            "errors": [],
        }
    
    # complete の場合、通常計算
    inp = PricingInput(
        purchase_price=normalized_data.get("purchase_price", 0.0),
        selling_price=normalized_data.get("selling_price", 0.0),
        shipping_cost=normalized_data.get("shipping_cost", 0.0),
        customs_duty=normalized_data.get("customs_duty", 0.0),
        procurement_fee=normalized_data.get("procurement_fee", 0.0),
        transaction_fee=normalized_data.get("transaction_fee", 0.0),
    )
    
    result = calculate_pricing(inp)
    
    return {
        "status": "complete",
        "revenue": result.revenue,
        "total_cost": result.total_cost,
        "profit": result.profit,
        "profit_rate": result.profit_rate,
        "errors": [],
    }

