# -*- coding: utf-8 -*-
"""
SelfHealingPatchAgent - Self-Healing による自動パッチ候補生成エージェント

CR-ATELIER-003 Phase D-6: failure_context + failure_analysis を入力として、
サイト設定 (site_config / overrides.local.json 相当) に対する
「自動パッチ候補」を JSON として生成・保存する。

このフェーズでは、パッチの「適用」は行わず、
候補生成と保存、および DiscoveryResult へのパス埋め込みまでに限定する。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from app.core.run_context import RunContext

logger = logging.getLogger(__name__)


class SelfHealingPatchAgent:
    """
    Self-Healing による自動パッチ候補生成エージェント
    
    failure_context と failure_analysis を元に、
    サイト設定に対するパッチ候補を生成・保存する。
    """
    
    def __init__(self, runtime_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """
        SelfHealingPatchAgent を初期化
        
        Args:
            runtime_kwargs: ランタイム設定（現在は未使用、将来の拡張用）
        """
        self.runtime_kwargs = runtime_kwargs or {}
        self.logger = logger
    
    async def build_patch_candidate(
        self,
        *,
        failure_context: Dict[str, Any],
        failure_analysis: Dict[str, Any],
        site_config: Dict[str, Any],
        run_context: RunContext,
        selector_repair_result: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        パッチ候補を生成・保存する
        
        Args:
            failure_context: 標準化された failure_context 辞書
            failure_analysis: FailureAnalysisAgent による分析結果
            site_config: サイト設定
            run_context: RunContext オブジェクト
            selector_repair_result: SelectorRepairAgent による selector repair 結果（オプション）
        
        Returns:
            生成したパッチ候補の辞書。生成に失敗した場合は None。
        """
        try:
            # パッチ候補を構築
            patch_candidate = self._build_patch_dict(
                failure_context=failure_context,
                failure_analysis=failure_analysis,
                site_config=site_config,
                run_context=run_context,
                selector_repair_result=selector_repair_result,
            )
            
            # パッチ候補を保存
            run_context.save_json("patch_candidate_self_healing.json", patch_candidate)
            self.logger.info(
                f"[SelfHealingPatch] Patch candidate generated and saved: "
                f"run_id={patch_candidate.get('run_id')}, "
                f"changes_count={len(patch_candidate.get('changes', []))}"
            )
            
            return patch_candidate
            
        except Exception as e:
            self.logger.error(
                f"[SelfHealingPatch] Failed to build patch candidate: {e}",
                exc_info=True
            )
            return None
    
    def _build_patch_dict(
        self,
        *,
        failure_context: Dict[str, Any],
        failure_analysis: Dict[str, Any],
        site_config: Dict[str, Any],
        run_context: RunContext,
        selector_repair_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        パッチ候補の辞書を構築する
        
        Args:
            failure_context: 標準化された failure_context 辞書
            failure_analysis: FailureAnalysisAgent による分析結果
            site_config: サイト設定
            run_context: RunContext オブジェクト
            selector_repair_result: SelectorRepairAgent による selector repair 結果（オプション）
        
        Returns:
            パッチ候補の辞書
        """
        site_code = failure_context.get("site", "unknown")
        run_id = failure_context.get("run_id", getattr(run_context, "run_id", "unknown"))
        error_type = failure_context.get("error_type", "unknown")
        error_message = failure_context.get("error_message", "")
        
        # 変更提案を生成
        changes = self._generate_changes(
            error_type=error_type,
            error_message=error_message,
            failure_analysis=failure_analysis,
            site_config=site_config,
        )
        
        # CR-ATELIER-003 Phase D-10: selector_repair_result から selector_patch を追加
        if selector_repair_result and selector_repair_result.get("candidates"):
            selector_changes = self._generate_selector_patches(
                selector_repair_result=selector_repair_result,
                site_config=site_config,
            )
            changes.extend(selector_changes)
            # strategy を更新
            strategy = "heuristic_v2_selector_aware"
        else:
            strategy = "heuristic_v1"
        
        # パッチ候補を構築
        patch_candidate = {
            "target_site": site_code,
            "run_id": run_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "strategy": strategy,
            "changes": changes,
            "notes": {
                "summary": failure_analysis.get("summary", ""),
                "root_causes": failure_analysis.get("root_causes", []),
                "suggested_fixes": failure_analysis.get("suggested_fixes", []),
            },
        }
        
        return patch_candidate
    
    def _generate_changes(
        self,
        *,
        error_type: str,
        error_message: str,
        failure_analysis: Dict[str, Any],
        site_config: Dict[str, Any],
    ) -> list[Dict[str, Any]]:
        """
        エラー情報から変更提案を生成する
        
        CR-ATELIER-003 Phase D-6: 簡易ルールベースの実装
        
        Args:
            error_type: エラータイプ
            error_message: エラーメッセージ
            failure_analysis: FailureAnalysisAgent による分析結果
            site_config: サイト設定
        
        Returns:
            変更提案のリスト
        """
        changes = []
        
        # ルール 1: navigation_failed または trap_recovery_failed の場合、timeout_sec を +30 秒
        if error_type in ("navigation_failed", "trap_recovery_failed"):
            current_timeout = site_config.get("discovery_settings", {}).get("timeout_sec", 60)
            changes.append({
                "path": "discovery_settings.timeout_sec",
                "action": "increase",
                "value_delta": 30,
                "current_value": current_timeout,
                "proposed_value": current_timeout + 30,
            })
        
        # ルール 2: error_message 内に "selector" または "not found" が含まれる場合、
        # pdp selector に対する「要再確認」フラグを付ける
        error_message_lower = error_message.lower()
        if "selector" in error_message_lower or "not found" in error_message_lower:
            changes.append({
                "path": "selectors.pdp",
                "action": "review_required",
                "reason": "Selector may be outdated or incorrect",
                "current_selectors": site_config.get("selectors", {}).get("pdp", {}),
            })
        
        # ルール 3: suggested_fixes に "timeout" が含まれる場合、timeout_sec を +30 秒
        suggested_fixes = failure_analysis.get("suggested_fixes", [])
        for fix in suggested_fixes:
            if isinstance(fix, str) and "timeout" in fix.lower():
                # 既に timeout_sec の変更が提案されている場合はスキップ
                if not any(c.get("path") == "discovery_settings.timeout_sec" for c in changes):
                    current_timeout = site_config.get("discovery_settings", {}).get("timeout_sec", 60)
                    changes.append({
                        "path": "discovery_settings.timeout_sec",
                        "action": "increase",
                        "value_delta": 30,
                        "current_value": current_timeout,
                        "proposed_value": current_timeout + 30,
                    })
                break
        
        return changes
    
    def _generate_selector_patches(
        self,
        *,
        selector_repair_result: Dict[str, Any],
        site_config: Dict[str, Any],
    ) -> list[Dict[str, Any]]:
        """
        CR-ATELIER-003 Phase D-10: SelectorRepairAgent の結果から selector_patch を生成する
        
        Args:
            selector_repair_result: SelectorRepairAgent.propose_selector_patches() の戻り値
            site_config: サイト設定
        
        Returns:
            selector_patch のリスト
        """
        changes = []
        page_type = selector_repair_result.get("page_type", "pdp")
        candidates = selector_repair_result.get("candidates", [])
        
        for candidate in candidates:
            target = candidate.get("target")
            old_selector = candidate.get("old_selector", "")
            new_selector = candidate.get("new_selector", "")
            confidence = candidate.get("confidence", 0.0)
            reason = candidate.get("reason", "")
            
            if not target or not new_selector:
                continue
            
            # 現行の selectors から old_selector を取得（指定されていない場合）
            if not old_selector:
                selectors = site_config.get("selectors", {}).get(page_type, {})
                if isinstance(selectors.get(target), list):
                    old_selector = selectors[target][0] if selectors[target] else ""
                elif isinstance(selectors.get(target), str):
                    old_selector = selectors[target]
            
            # selector_patch を追加
            changes.append({
                "path": f"selectors.{page_type}.{target}",
                "action": "selector_patch",
                "old_value": old_selector,
                "new_value": new_selector,
                "confidence": confidence,
                "reason": reason,
            })
        
        return changes
