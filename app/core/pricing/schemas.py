from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PricingInput:
    """
    利益計算の入力。Flask Product モデルと1対1で対応させる。
    """
    purchase_price: float          # 仕入れ原価（original_currency建て）
    selling_price: float           # 販売価格（BUYMA出品価格、JPY）
    transaction_fee: float = 0.0   # 固定の決済手数料など
    shipping_cost: float = 0.0     # 国内送料（転送倉庫→日本）
    customs_duty: float = 0.0      # 関税（0なら自動計算、>0なら手動入力を優先）
    procurement_fee: float = 0.0   # 代行手数料・その他経費
    warehouse_shipping_cost: float = 0.0   # 転送倉庫送料（海外→転送倉庫）
    original_currency: str = "JPY"         # 仕入れ通貨
    exchange_rate: float = 1.0             # 適用為替レート（1単位あたりJPY）
    item_category: str = ""                # 品目カテゴリ（関税率自動決定用）
    item_material: str = ""                # 素材（関税率自動決定用）


@dataclass
class PricingResult:
    """
    計算結果。UI / API からはこの型を経由して読む。
    """
    revenue: float                # 売上 (selling_price)
    total_cost: float             # 総コスト
    profit: float                 # 利益
    profit_rate: float            # 利益率 (profit / revenue)
    purchase_price_jpy: float = 0.0   # JPY換算仕入れ原価
    total_shipping_cost: float = 0.0  # 送料合計（国内+転送倉庫）
    auto_customs_duty: float = 0.0    # 自動計算された関税
    customs_rate_used: float = 0.0    # 適用された関税率
