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

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TypedDict


class Evidence(TypedDict, total=False):
    """DiscoveryResult.evidence の型定義。全フィールドOptional。"""

    # --- 正常系 ---
    extracted_data: dict[str, Any]
    extracted_fields: list[str]
    screenshots: list[str]
    dom_snapshots: list[str]
    final_url: str
    run_id: str
    success_stage: str
    criteria: dict[str, Any]
    urls: dict[str, str]
    link_collection: dict[str, Any]
    url: str

    # --- Self-Healing ---
    self_healing_attempts: int
    auto_patches_applied: list[str]
    patch_backups: list[str]
    self_healing_patch_candidate: dict[str, Any]
    selectors_update: dict[str, Any] | None
    code_patch: str
    repair_log: list[Any]
    steps_taken: int | None
    learned_selectors: dict[str, Any]

    # --- 異常系 ---
    failure_context: dict[str, Any]
    failure_analysis: dict[str, Any]
    error: str
    status: str
    initial_failure: dict[str, Any]
    message: str
    plp_result: Any

    # --- 設定 ---
    final_config: dict[str, Any]
    timeout_sec: int


# LLM応答の共通結果
@dataclass(kw_only=True, slots=True)
class GenerateResult:
    """LLMからの生成結果を格納する標準データクラス。"""

    text: str
    tokens: dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    model_family: str = "unknown"
    cached: bool = False
    sentiment: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"] = "NEUTRAL"

    def to_dict(self) -> dict[str, Any]:
        """このオブジェクトを辞書に変換します。"""
        return asdict(self)


# セレクタ発見や偵察結果の共通モデル
@dataclass(kw_only=True, slots=True)
class DiscoveryResult:
    """エージェントによる発見・偵察タスクの結果を格納する標準データクラス。"""

    ok: bool
    site: str
    query: str
    proposal: dict[str, Any] = field(default_factory=dict)
    evidence: Evidence = field(default_factory=Evidence)  # type: ignore[misc]
    ai_analysis: GenerateResult | None = None
    message: str | None = None
    screenshot: str | None = None
    file_path: str | None = None  # 生成ファイルの保存先

    def to_dict(self) -> dict[str, Any]:
        """このオブジェクトを辞書に変換します。ai_analysisも再帰的に変換します。"""
        data = asdict(self)
        if self.ai_analysis:
            data["ai_analysis"] = self.ai_analysis.to_dict()
        return data
