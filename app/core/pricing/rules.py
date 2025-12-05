from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Dict


@dataclass
class PricingConfig:
    """
    BUYMA 用の料金ルール。
    
    - buyma_platform_fee_rate: プラットフォーム販売手数料（デフォルト 7.7%）
      → 市場分析・シミュレーション用の純粋なプラットフォーム手数料
    
    - buyma_effective_fee_rate: 実効的な総手数料（デフォルト 14.2%）
      → 販売手数料 + 決済手数料 + その他を含む実運用での総負担率
      → Product.calculate_profit() などで使用
    
    - additional_fee_rate: その他の追加手数料（オプション）
    """
    buyma_platform_fee_rate: float = 0.077   # 7.7% - プラットフォーム販売手数料
    buyma_effective_fee_rate: float = 0.142  # 14.2% - 実効総手数料
    additional_fee_rate: float = 0.0         # その他の手数料


_DEFAULT_CONFIG = PricingConfig(
    buyma_platform_fee_rate=0.077,
    buyma_effective_fee_rate=0.142,
    additional_fee_rate=0.0,
)


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
        )
    except Exception:
        # 壊れた設定ファイルは無視してデフォルトへフォールバック
        return _DEFAULT_CONFIG

