from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


def nz(value: float | int | None) -> float:
    """None を 0 に変換するヘルパー（価格計算用）。"""
    return value if value is not None else 0.0


class PriceSource(str, Enum):
    """価格データの取得元 — 信頼度に応じて計可/不可を判定。"""

    BROWSER_VERIFIED = "browser_verified"  # ブラウザ/スクレイパで実際のページから取得
    MANUAL_INPUT = "manual_input"  # 人間が手動入力
    API_VERIFIED = "api_verified"  # 公式API等から取得
    KEYWORD_GUESS = "keyword_guess"  # キーワードマッチングによる推測（禁止）
    UNKNOWN = "unknown"  # 出所不明（禁止）


@dataclass
class PriceData:
    """価格データは必ず出所を明示する。推測データは利益計算に使えない。"""

    price: float
    source: PriceSource = PriceSource.UNKNOWN
    label: str = ""  # 例: "SSENSE $261", "BUYMA Black Lettering ¥45,600"

    def is_reliable(self) -> bool:
        """利益計算に使える信頼性かどうか。"""
        return self.source in (
            PriceSource.BROWSER_VERIFIED,
            PriceSource.MANUAL_INPUT,
            PriceSource.API_VERIFIED,
        )


class PriceIntegrityError(ValueError):
    """推測/未確認データでの利益計算を防止する例外。"""


@dataclass
class PricingInput:
    """
    利益計算の入力。Flask Product モデルと1対1で対応させる。
    """

    purchase_price: float  # 仕入れ原価（original_currency建て）
    selling_price: float  # 販売価格（BUYMA出品価格、JPY）
    transaction_fee: float = 0.0  # 固定の決済手数料など
    shipping_cost: float = 0.0  # 国内送料（転送倉庫→日本）
    customs_duty: float = 0.0  # 関税（0なら自動計算、>0なら手動入力を優先）
    procurement_fee: float = 0.0  # 代行手数料・その他経費
    warehouse_shipping_cost: float = 0.0  # 転送倉庫送料（海外→転送倉庫）
    original_currency: str = "JPY"  # 仕入れ通貨
    exchange_rate: float = 1.0  # 適用為替レート（1単位あたりJPY）
    item_category: str = ""  # 品目カテゴリ（関税率自動決定用）
    item_material: str = ""  # 素材（関税率自動決定用）

    # --- データ信頼性チェック ---
    # Phase1(ISSUE-101/102): デフォルト UNKNOWN を廃止し None 許容に変更。
    # None は「未設定」を意味し、validate_sources で UNKNOWN 扱い（信頼できない→例外）。
    # 6経路は Phase1 で一時的に skip_source_validation=True で動作維持→Phase2 で source 明示。
    purchase_price_source: PriceSource | None = None
    selling_price_source: PriceSource | None = None

    def validate_sources(self) -> None:
        """推測/未確認データで計算しようとしたら例外を投げる。

        None（未設定）は UNKNOWN 扱い（信頼できない→例外）。
        """
        errors: list[str] = []
        psrc = self.purchase_price_source or PriceSource.UNKNOWN
        ssrc = self.selling_price_source or PriceSource.UNKNOWN
        if not PriceData(self.purchase_price, psrc).is_reliable():
            errors.append(
                f"仕入価格のデータソースが信頼できません: {psrc.value}。"
                "ブラウザ/スクレイパで実際の価格を確認してください。"
            )
        if not PriceData(self.selling_price, ssrc).is_reliable():
            errors.append(
                f"販売価格のデータソースが信頼できません: {ssrc.value}。"
                "BUYMAで実際の出品価格を確認してください。"
            )
        if errors:
            raise PriceIntegrityError("\n".join(errors))


@dataclass
class PricingResult:
    """
    計算結果。UI / API からはこの型を経由して読む。
    """

    revenue: float  # 売上 (selling_price)
    total_cost: float  # 総コスト
    profit: float  # 利益
    profit_rate: float  # 利益率 (profit / revenue)
    purchase_price_jpy: float = 0.0  # JPY換算仕入れ原価
    total_shipping_cost: float = 0.0  # 送料合計（国内+転送倉庫）
    auto_customs_duty: float = 0.0  # 自動計算された関税
    customs_rate_used: float = 0.0  # 適用された関税率
