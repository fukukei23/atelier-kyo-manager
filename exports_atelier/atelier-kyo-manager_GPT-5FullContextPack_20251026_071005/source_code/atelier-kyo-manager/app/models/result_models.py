# ==============================================================================
# File: result_models.py
# Registry: C:\Users\USER\tools\atelier-kyo-manager\app\models\result_models.py
# Date & Time (JST): 2025-09-16 13:51:00
# Version: 1.0J (Initial Release)
#
# --- What's New (v1.0J) ---
#  - [Initial Release] Created standardized data structures for use across the
#    entire multi-agent system.
#  - `GenerateResult`: A robust model for LLM responses, ensuring consistency.
#  - `DiscoveryResult`: A standardized model for scouting and discovery tasks.
#  - Utilizes modern Python features (`kw_only=True`, `slots=True`) for enhanced
#    safety and performance.
#
# --- How To Use ---
# Import these dataclasses into any agent or utility that needs to handle
# standardized results.
#
# from app.models.result_models import GenerateResult, DiscoveryResult
# ==============================================================================
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Literal, Optional

# LLM応答の共通結果
@dataclass(kw_only=True, slots=True)
class GenerateResult:
    """LLMからの生成結果を格納する標準データクラス。"""
    text: str
    tokens: Dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    model_family: str = "unknown"
    cached: bool = False
    sentiment: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"] = "NEUTRAL"

    def to_dict(self) -> Dict[str, Any]:
        """このオブジェクトを辞書に変換します。"""
        return asdict(self)

# セレクタ発見や偵察結果の共通モデル
@dataclass(kw_only=True, slots=True)
class DiscoveryResult:
    """エージェントによる発見・偵察タスクの結果を格納する標準データクラス。"""
    ok: bool
    site: str
    query: str
    proposal: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    ai_analysis: Optional[GenerateResult] = None
    message: Optional[str] = None
    screenshot: Optional[str] = None
    file_path: Optional[str] = None  # 生成ファイルの保存先

    def to_dict(self) -> Dict[str, Any]:
        """このオブジェクトを辞書に変換します。ai_analysisも再帰的に変換します。"""
        data = asdict(self)
        if self.ai_analysis:
            data['ai_analysis'] = self.ai_analysis.to_dict()
        return data
