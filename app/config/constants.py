from __future__ import annotations

"""アプリケーション全体で使用するビジネス定数（SSOT）。"""

# BUYMA 手数料
DOMESTIC_COMMISSION_RATE: float = 0.077   # 7.7% — 国内仕入れ時の成約手数料
OVERSEAS_COMMISSION_RATE: float = 0.055   # 5.5% — 海外仕入れ時の成約手数料
PLATFORM_FEE_RATE: float = 0.077          # 7.7% — プラットフォーム販売手数料
EFFECTIVE_FEE_RATE: float = 0.142         # 14.2% — 実効総手数料

# 振込手数料
TRANSFER_FEE: float = 220.0               # 振込手数料（楽天銀行想定）

# 為替フォールバック
DEFAULT_EXCHANGE_RATE_USDJPY: float = 150.0

# 決済方法別 延長期限（日数）
PAYMENT_METHOD_EXTENSION_DAYS: dict[str, int] = {
    "credit_card": 45,
    "rakuten_pay": 45,
    "d_pay": 25,
    "au_pay": 25,
    "paidy": 25,
    "bank_transfer": 90,
    "convenience": 90,
    "paypay": 90,
    "amazon_pay": 90,
}

# 仕入区分別 平均着金日数
EXPECTED_PAYMENT_DAYS: dict[str, int] = {
    "domestic": 15,
    "overseas": 25,
}
