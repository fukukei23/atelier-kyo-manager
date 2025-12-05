from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PricingInput:
    """
    利益計算の入力。Flask Product モデルと1対1で対応させる。
    """
    purchase_price: float          # 仕入れ原価
    selling_price: float           # 販売価格（BUYMA出品価格）
    transaction_fee: float = 0.0   # 固定の決済手数料など（% ではなく金額で渡す想定）
    shipping_cost: float = 0.0     # 送料
    customs_duty: float = 0.0      # 関税
    procurement_fee: float = 0.0   # 代行手数料・その他経費


@dataclass
class PricingResult:
    """
    計算結果。UI / API からはこの型を経由して読む。
    """
    revenue: float        # 売上 (selling_price)
    total_cost: float     # 総コスト
    profit: float         # 利益
    profit_rate: float    # 利益率 (profit / revenue)

