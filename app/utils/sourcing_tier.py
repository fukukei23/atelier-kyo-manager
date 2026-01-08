"""Tier判定を行うモジュール。

利益計算結果に基づいて Tier A/B/C/D を判定する。
"""
from __future__ import annotations

from typing import Any, Dict, Literal


def judge_tier(profitability_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    利益計算結果から Tier を判定する。
    
    Args:
        profitability_result: calculate_profitability の結果
        
    Returns:
        {
            "tier": "A" | "B" | "C" | "D",
            "reason": str
        }
    """
    status = profitability_result.get("status")
    
    # invalid の場合は Tier D
    if status == "invalid":
        return {
            "tier": "D",
            "reason": "入力データが無効です",
        }
    
    # partial の場合は最大 Tier B（A禁止）
    if status == "partial":
        profit_rate = profitability_result.get("profit_rate")
        if profit_rate is None:
            return {
                "tier": "C",
                "reason": "partial ステータスで利益率が計算できませんでした",
            }
        
        # partial でも利益率が高い場合は B、低い場合は C
        if profit_rate >= 0.20:  # 20%以上
            return {
                "tier": "B",
                "reason": "partial ステータスですが利益率が高いため Tier B",
            }
        else:
            return {
                "tier": "C",
                "reason": "partial ステータスで利益率が低いため Tier C",
            }
    
    # complete の場合、利益率に基づいて判定
    profit_rate = profitability_result.get("profit_rate")
    if profit_rate is None:
        return {
            "tier": "D",
            "reason": "利益率が計算できませんでした",
        }
    
    # Tier A: 利益率 30%以上
    if profit_rate >= 0.30:
        return {
            "tier": "A",
            "reason": f"利益率 {profit_rate:.2%} が高いため Tier A",
        }
    
    # Tier B: 利益率 20%以上30%未満
    if profit_rate >= 0.20:
        return {
            "tier": "B",
            "reason": f"利益率 {profit_rate:.2%} が中程度のため Tier B",
        }
    
    # Tier C: 利益率 20%未満
    return {
        "tier": "C",
        "reason": f"利益率 {profit_rate:.2%} が低いため Tier C",
    }

