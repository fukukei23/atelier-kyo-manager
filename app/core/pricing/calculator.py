from __future__ import annotations

from .rules import PricingConfig, load_pricing_config, resolve_customs_rate
from .schemas import PricingInput, PricingResult


def calculate_pricing(
    inp: PricingInput,
    config: PricingConfig | None = None,
    source_type: str = "domestic",
    *,
    skip_source_validation: bool = False,
    allow_estimated: bool = False,
) -> PricingResult:
    """
    利益計算のコアロジック。
    Flask/CLI/API からはこの関数だけを呼ぶ。

    source_type: "domestic" or "overseas" — 成約手数料率の判定に使用
    skip_source_validation: True の場合、データソース検証をスキップ（後方互換用）
    allow_estimated: True の場合のみ ESTIMATED source を許容（結果は estimate_mark=True）
    """
    # 0. データソース信頼性チェック — 推測データでの計算を防止
    estimate_mark = False
    if not skip_source_validation:
        estimate_mark = inp.validate_sources(allow_estimated=allow_estimated)

    cfg = config or load_pricing_config()

    # 1. 為替変換: original_currency != "JPY" の場合に purchase_price を JPY 換算
    if inp.original_currency != "JPY" and inp.exchange_rate > 0:
        purchase_price_jpy = inp.purchase_price * inp.exchange_rate
    else:
        purchase_price_jpy = inp.purchase_price

    # 2. 送料合計（国内送料 + 転送倉庫送料）
    total_shipping_cost = inp.shipping_cost + inp.warehouse_shipping_cost

    # 3. 関税自動計算（手動入力優先）
    if inp.customs_duty > 0:
        # 手動入力値をそのまま使用
        customs_duty = inp.customs_duty
        customs_rate_used = 0.0
        auto_customs_duty = 0.0
    else:
        # 自動計算: (JPY換算原価 + 転送倉庫送料) × 関税率
        customs_rate_used = resolve_customs_rate(inp.item_category, inp.item_material)
        auto_customs_duty = (purchase_price_jpy + inp.warehouse_shipping_cost) * customs_rate_used
        customs_duty = auto_customs_duty

    # 4. 成約手数料（source_type で国内/海外を切替）
    if source_type == "overseas":
        commission_fee = inp.selling_price * cfg.overseas_commission_rate
    else:
        commission_fee = inp.selling_price * cfg.domestic_commission_rate

    # 5. その他手数料
    additional_fee = inp.selling_price * cfg.additional_fee_rate

    # 6. 総コスト
    total_cost = (
        purchase_price_jpy
        + total_shipping_cost
        + customs_duty
        + inp.procurement_fee
        + inp.transaction_fee
        + commission_fee
        + additional_fee
        + cfg.transfer_fee
    )

    # 7. 利益と利益率
    revenue = inp.selling_price
    profit = revenue - total_cost
    profit_rate = (profit / revenue) if revenue != 0 else 0.0

    return PricingResult(
        revenue=round(revenue, 2),
        total_cost=round(total_cost, 2),
        profit=round(profit, 2),
        profit_rate=round(profit_rate, 4),
        purchase_price_jpy=round(purchase_price_jpy, 2),
        total_shipping_cost=round(total_shipping_cost, 2),
        auto_customs_duty=round(auto_customs_duty, 2),
        customs_rate_used=round(customs_rate_used, 4),
        estimate_mark=estimate_mark,
    )
