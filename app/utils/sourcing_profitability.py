"""利益計算を行うモジュール。

既存の app.core.pricing.calculator を使用して利益を計算する。
partial（unknown含む）の場合は安全側に処理する。
"""

from __future__ import annotations

from typing import Any

from app.core.pricing.calculator import calculate_pricing
from app.core.pricing.rules import load_pricing_config
from app.core.pricing.schemas import PriceSource, PricingInput


def calculate_profitability(normalized_data: dict[str, Any] | None) -> dict[str, Any]:
    """
    正規化された入力データから利益を計算する。

    Args:
        normalized_data: validate_sourcing_input の normalized 結果

    Returns:
        {
            "status": "complete" | "partial" | "invalid",
            "revenue": float | None,
            "total_cost": float | None,  # complete の場合のみ
            "known_cost_sum": float | None,  # partial の場合のみ
            "profit": float | None,
            "profit_rate": float | None,
            "profit_upper_bound": float | None,  # partial の場合のみ（上限）
            "missing_fields": List[str] | None,  # partial の場合のみ
            "errors": List[str]
        }
    """
    if normalized_data is None:
        return {
            "status": "invalid",
            "revenue": None,
            "total_cost": None,
            "known_cost_sum": None,
            "profit": None,
            "profit_rate": None,
            "profit_upper_bound": None,
            "missing_fields": None,
            "errors": ["入力データが無効です"],
        }

    # unknown が含まれている場合は partial
    missing_fields: list[str] = []
    for key, value in normalized_data.items():
        if value is None:
            missing_fields.append(key)
    has_unknown = len(missing_fields) > 0

    if has_unknown:
        # partial の場合、計算可能な項目のみで計算
        # ただし、必須項目（purchase_price, selling_price）が None の場合は計算不可
        if normalized_data.get("purchase_price") is None or normalized_data.get("selling_price") is None:
            return {
                "status": "partial",
                "revenue": None,
                "total_cost": None,
                "known_cost_sum": None,
                "profit": None,
                "profit_rate": None,
                "profit_upper_bound": None,
                "missing_fields": missing_fields,
                "errors": ["必須項目が 'unknown' のため計算できません"],
            }

        # revenue（selling_price）を確定
        selling_price = normalized_data.get("selling_price", 0.0)
        revenue = float(selling_price)

        # known_cost_sum（確定コスト合計）を計算
        # unknown の項目（None）は合計に含めず、確定している項目のみを合計
        cfg = load_pricing_config()
        buyma_fee = revenue * cfg.buyma_effective_fee_rate
        additional_fee = revenue * cfg.additional_fee_rate

        # 確定している項目のみを合計（None の項目は合計から除外）
        known_cost_sum = 0.0
        if normalized_data.get("purchase_price") is not None:
            known_cost_sum += float(normalized_data["purchase_price"])
        if normalized_data.get("shipping_cost") is not None:
            known_cost_sum += float(normalized_data["shipping_cost"])
        if normalized_data.get("customs_duty") is not None:
            known_cost_sum += float(normalized_data["customs_duty"])
        if normalized_data.get("procurement_fee") is not None:
            known_cost_sum += float(normalized_data["procurement_fee"])
        if normalized_data.get("transaction_fee") is not None:
            known_cost_sum += float(normalized_data["transaction_fee"])
        if normalized_data.get("warehouse_shipping_cost") is not None:
            known_cost_sum += float(normalized_data["warehouse_shipping_cost"])
        known_cost_sum += buyma_fee
        known_cost_sum += additional_fee

        # profit_upper_bound = revenue - known_cost_sum（上限であり確定利益ではない）
        profit_upper_bound = revenue - known_cost_sum

        return {
            "status": "partial",
            "revenue": revenue,
            "total_cost": None,
            "known_cost_sum": round(known_cost_sum, 2),
            "profit": None,  # 確定できないため null
            "profit_rate": None,  # 確定できないため null
            "profit_upper_bound": round(profit_upper_bound, 2),
            "missing_fields": missing_fields,
            "errors": [],
        }

    # complete の場合、通常計算
    # 出所: 入力JSON/CSVの source カラム（任意）・未指定は人手提供扱い manual_input（T7）
    inp = PricingInput(
        purchase_price=normalized_data.get("purchase_price", 0.0),
        selling_price=normalized_data.get("selling_price", 0.0),
        shipping_cost=normalized_data.get("shipping_cost", 0.0),
        customs_duty=normalized_data.get("customs_duty", 0.0),
        procurement_fee=normalized_data.get("procurement_fee", 0.0),
        transaction_fee=normalized_data.get("transaction_fee", 0.0),
        warehouse_shipping_cost=normalized_data.get("warehouse_shipping_cost", 0.0),
        original_currency=normalized_data.get("original_currency", "JPY"),
        exchange_rate=normalized_data.get("exchange_rate", 1.0),
        item_category=normalized_data.get("item_category", ""),
        item_material=normalized_data.get("item_material", ""),
        purchase_price_source=PriceSource(normalized_data.get("purchase_price_source", "manual_input")),
        selling_price_source=PriceSource(normalized_data.get("selling_price_source", "manual_input")),
    )

    # 推定（estimated）宣言は計算するが profit_status で区別（T9 以降の発注ブロック用）
    result = calculate_pricing(inp, allow_estimated=True)

    return {
        "status": "complete",
        "revenue": result.revenue,
        "total_cost": result.total_cost,
        "known_cost_sum": None,
        "profit": result.profit,
        "profit_rate": result.profit_rate,
        "profit_upper_bound": None,
        "missing_fields": None,
        "errors": [],
        "profit_status": "estimated" if result.estimate_mark else "confirmed",
    }
