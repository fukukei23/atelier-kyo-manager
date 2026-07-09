from __future__ import annotations

"""仕入れコスト計算用の定数テーブル。

経営者判断ファイル（docs/経営者判断.md）のセクション3に基づく。
手数料・関税率は既存の app/core/pricing/rules.py, app/config/constants.py を再利用し、
ここでは転送倉庫送料のみを管理する。

出典: Buyandship公式（2026-04-27改定、2026-05-25確認）
  最初の3lb（~1.36kg）: ¥3,300
  追加1lb（~0.45kg）: ¥900
"""

# Buyandship 送料テーブル（EUR → 日本）
# 単位: JPY。商品カテゴリに基づく概算。
BUYANDSHIP_SHIPPING_RATES: dict[str, float] = {
    "light": 3300.0,  # ~1lb（財布・小物アクセサリー）= 3,300円（3lb以下固定）
    "medium": 3300.0,  # ~2lb（バッグ・靴）= 3,300円（3lb以下固定）
    "heavy": 4200.0,  # ~4lb（大型バッグ・コート）= 3,300 + 900
    "default": 3300.0,  # 不明時のデフォルト
}

# 商品カテゴリ → Buyandship重量帯のマッピング
CATEGORY_WEIGHT_CLASS: dict[str, str] = {
    "wallet": "light",
    "accessory": "light",
    "watch": "light",
    "bag": "medium",
    "shoes": "medium",
    "clothing": "medium",
    "leather": "medium",
}

# BUYMA想定販売価格のデフォルトマークアップ率
# 経営者判断: 現状1.5倍で利益率11-14%。やや高めの1.3倍をデフォルトとする。
DEFAULT_MARKUP_RATE = 1.3


def get_buyandship_shipping(category: str | None = None) -> float:
    """カテゴリに基づくBuyandship送料を返す。"""
    weight_class = CATEGORY_WEIGHT_CLASS.get(category or "", "default")
    return BUYANDSHIP_SHIPPING_RATES.get(weight_class, BUYANDSHIP_SHIPPING_RATES["default"])
