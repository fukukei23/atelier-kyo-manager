from __future__ import annotations

from typing import Optional

from .schemas import PricingInput, PricingResult
from .rules import PricingConfig, load_pricing_config


def calculate_pricing(
    inp: PricingInput,
    config: Optional[PricingConfig] = None,
) -> PricingResult:
    """
    利益計算のコアロジック。
    Flask/CLI/API からはこの関数だけを呼ぶ。
    """
    cfg = config or load_pricing_config()

    # 1. 手数料 (率指定の場合)
    # buyma_effective_fee_rate: 実運用での総手数料（14.2%）
    buyma_fee = inp.selling_price * cfg.buyma_effective_fee_rate
    additional_fee = inp.selling_price * cfg.additional_fee_rate

    # 2. 総コスト
    total_cost = (
        inp.purchase_price
        + inp.shipping_cost
        + inp.customs_duty
        + inp.procurement_fee
        + inp.transaction_fee
        + buyma_fee
        + additional_fee
    )

    # 3. 利益と利益率
    revenue = inp.selling_price
    profit = revenue - total_cost
    profit_rate = (profit / revenue) if revenue != 0 else 0.0

    return PricingResult(
        revenue=round(revenue, 2),
        total_cost=round(total_cost, 2),
        profit=round(profit, 2),
        profit_rate=round(profit_rate, 4),
    )

