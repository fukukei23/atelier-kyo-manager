from __future__ import annotations

"""アプリケーション全体で使用するビジネス定数（SSOT）。"""

# BUYMA 手数料
DOMESTIC_COMMISSION_RATE: float = 0.077  # 7.7% — 国内仕入れ時の成約手数料
OVERSEAS_COMMISSION_RATE: float = 0.055  # 5.5% — 海外仕入れ時の成約手数料
PLATFORM_FEE_RATE: float = 0.077  # 7.7% — プラットフォーム販売手数料
EFFECTIVE_FEE_RATE: float = 0.142  # 14.2% — 実効総手数料

# 振込手数料
TRANSFER_FEE: float = 220.0  # 振込手数料（楽天銀行想定）

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

# ---- 技術定数（インフラ・キャッシュ・リトライ） ----

# ChatBot 入力制限
MAX_SUBJECT_LENGTH: int = 200
MAX_MESSAGE_LENGTH: int = 1000

# PriceScraper
SCRAPER_CACHE_SIZE: int = 100
SCRAPER_REQUEST_TIMEOUT: int = 10
SCRAPER_MAX_RETRIES: int = 3

# 為替ユーティリティ
FX_REQUEST_TIMEOUT: int = 10
FX_CACHE_TTL_HOURS: int = 12

# LLM コントローラ
LLM_RETRY_BACKOFF_BASE: float = 1.5
LLM_LOCAL_CTX_SIZE: int = 2048
LLM_LOCAL_MAX_TOKENS: int = 512
LLM_LOCAL_TEMPERATURE: float = 0.3
LLM_CACHE_TTL_SECONDS: int = 2_592_000  # 30日
CHAT_HISTORY_RETENTION_DAYS: int = 7
