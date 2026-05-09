from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Dict

from app.config.constants import (
    DOMESTIC_COMMISSION_RATE,
    OVERSEAS_COMMISSION_RATE,
    PLATFORM_FEE_RATE,
    EFFECTIVE_FEE_RATE,
    TRANSFER_FEE,
)


# 品目別関税率テーブル
CUSTOMS_RATE_TABLE: dict[str, float] = {
    "leather": 0.12,        # 革製品: 12%
    "bag": 0.11,            # バッグ: 11%
    "shoes": 0.11,          # 靴: 11%
    "wallet": 0.12,         # 財布(革): 12%
    "watch": 0.10,          # 腕時計: 10%
    "accessory": 0.10,      # アクセサリー: 10%
    "clothing": 0.10,       # 衣類: 10%
    "default": 0.10,        # デフォルト: 10%
}


def resolve_customs_rate(category: str | None, material: str | None) -> float:
    """品目カテゴリと素材から関税率を決定する。"""
    # 素材ベースの判定（優先度高）
    if material:
        mat_lower = material.lower()
        if any(m in mat_lower for m in ["leather", "革", "レザー"]):
            return CUSTOMS_RATE_TABLE["leather"]
    # カテゴリベースの判定
    if category:
        cat_lower = category.lower()
        if any(c in cat_lower for c in ["bag", "バッグ"]):
            return CUSTOMS_RATE_TABLE["bag"]
        if any(c in cat_lower for c in ["shoes", "靴", "シューズ"]):
            return CUSTOMS_RATE_TABLE["shoes"]
        if any(c in cat_lower for c in ["wallet", "財布"]):
            return CUSTOMS_RATE_TABLE["wallet"]
        if any(c in cat_lower for c in ["watch", "腕時計"]):
            return CUSTOMS_RATE_TABLE["watch"]
    return CUSTOMS_RATE_TABLE["default"]


@dataclass
class PricingConfig:
    """
    BUYMA 用の料金ルール。

    - buyma_platform_fee_rate: プラットフォーム販売手数料（デフォルト 7.7%）
      → 市場分析・シミュレーション用の純粋なプラットフォーム手数料

    - buyma_effective_fee_rate: 実効的な総手数料（デフォルト 14.2%）
      → 販売手数料 + 決済手数料 + その他を含む実運用での総負担率
      → calculator.py で使用

    - additional_fee_rate: その他の追加手数料（オプション）

    - domestic_commission_rate: 国内仕入れ時の成約手数料率
    - overseas_commission_rate: 海外仕入れ時の成約手数料率
    - transfer_fee: 振込手数料（固定）
    """
    buyma_platform_fee_rate: float = PLATFORM_FEE_RATE
    buyma_effective_fee_rate: float = EFFECTIVE_FEE_RATE
    additional_fee_rate: float = 0.0
    domestic_commission_rate: float = DOMESTIC_COMMISSION_RATE
    overseas_commission_rate: float = OVERSEAS_COMMISSION_RATE
    transfer_fee: float = TRANSFER_FEE


_DEFAULT_CONFIG = PricingConfig()


def load_pricing_config(config_path: str | None = None) -> PricingConfig:
    """
    JSON から設定を読み込む。
    path 未指定 or 読み込み失敗時はデフォルト値を返す。

    後方互換性:
    - 'buyma_fee_rate' が存在する場合は 'buyma_effective_fee_rate' として扱う
    """
    if not config_path:
        return _DEFAULT_CONFIG

    p = Path(config_path)
    if not p.exists():
        return _DEFAULT_CONFIG

    try:
        data: Dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))

        # 後方互換性: buyma_fee_rate -> buyma_effective_fee_rate
        effective_rate = data.get(
            "buyma_effective_fee_rate",
            data.get("buyma_fee_rate", _DEFAULT_CONFIG.buyma_effective_fee_rate)
        )

        return PricingConfig(
            buyma_platform_fee_rate=float(
                data.get("buyma_platform_fee_rate", _DEFAULT_CONFIG.buyma_platform_fee_rate)
            ),
            buyma_effective_fee_rate=float(effective_rate),
            additional_fee_rate=float(
                data.get("additional_fee_rate", _DEFAULT_CONFIG.additional_fee_rate)
            ),
            domestic_commission_rate=float(
                data.get("domestic_commission_rate", _DEFAULT_CONFIG.domestic_commission_rate)
            ),
            overseas_commission_rate=float(
                data.get("overseas_commission_rate", _DEFAULT_CONFIG.overseas_commission_rate)
            ),
            transfer_fee=float(
                data.get("transfer_fee", _DEFAULT_CONFIG.transfer_fee)
            ),
        )
    except Exception:
        # 壊れた設定ファイルは無視してデフォルトへフォールバック
        return _DEFAULT_CONFIG
