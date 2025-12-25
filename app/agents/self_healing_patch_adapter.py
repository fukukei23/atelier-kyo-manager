# -*- coding: utf-8 -*-
"""
SelfHealingPatchAdapter - Self-Healing パッチ候補を diff 形式に変換するアダプタ

CR-ATELIER-003 Phase D-7: patch_candidate_self_healing.json の schema を、
overrides.local.json に適用可能な diff 形式へ変換する。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class SelfHealingPatchAdapter:
    """
    Self-Healing パッチ候補を diff 形式に変換するアダプタ
    
    Phase D-6 で生成された patch_candidate_self_healing.json の schema を、
    overrides.local.json に適用可能な diff 形式へ変換する。
    """
    
    def to_diff(
        self,
        *,
        patch_candidate: Dict[str, Any],
        site_config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        SelfHealingPatchAgent が生成した patch_candidate を
        patch_applier 用の diff リストに変換して返す。
        
        Args:
            patch_candidate: Phase D-6 で生成された patch_candidate 辞書
            site_config: 現行の site_config（MONCLER_OFFICIAL ブロックなど）
        
        Returns:
            diff 操作のリスト。各要素は {"op": "...", "path": "...", "value": ...} 形式
        """
        changes = patch_candidate.get("changes", [])
        if not changes:
            return []
        
        diff_ops = []
        
        for change in changes:
            path = change.get("path", "")
            action = change.get("action", "")
            
            if not path or not action:
                logger.warning(f"[PatchAdapter] Invalid change entry: {path=}, {action=}")
                continue
            
            # path を JSON Pointer 形式に変換（例: "discovery_settings.timeout_sec" -> "/discovery_settings/timeout_sec"）
            json_path = self._to_json_pointer(path)
            
            if action == "increase":
                # timeout_sec などの数値増加
                current_value = change.get("current_value")
                value_delta = change.get("value_delta", 0)
                
                if current_value is None:
                    # site_config から現在の値を取得
                    current_value = self._get_nested_value(site_config, path, 0)
                
                new_value = current_value + value_delta
                
                diff_ops.append({
                    "op": "replace",
                    "path": json_path,
                    "value": new_value,
                })
                
            elif action == "review_required":
                # selector 要再確認フラグ
                # __meta_flags というメタデータフィールドにフラグを設定
                meta_path = json_path.rstrip("/") + "/__meta_flags/need_review"
                
                diff_ops.append({
                    "op": "add",
                    "path": meta_path,
                    "value": True,
                })
                
            elif action == "set":
                # 値を直接設定
                value = change.get("value")
                if value is not None:
                    diff_ops.append({
                        "op": "replace" if self._path_exists(site_config, path) else "add",
                        "path": json_path,
                        "value": value,
                    })
            
            elif action == "selector_patch":
                # CR-ATELIER-003 Phase D-10: selector_patch を replace にマッピング
                new_value = change.get("new_value")
                old_value = change.get("old_value", "")
                confidence = change.get("confidence", 0.0)
                
                if new_value is None:
                    logger.warning(f"[PatchAdapter] selector_patch missing new_value for path: {path}")
                    continue
                
                # selector は通常リスト形式で保存されるため、リストに変換
                # 既存の値がリストの場合は、先頭に new_value を追加
                current_value = self._get_nested_value(site_config, path, [])
                if isinstance(current_value, list):
                    # 既存リストの先頭に new_value を追加（重複を避ける）
                    new_list = [new_value]
                    for item in current_value:
                        if item != new_value:
                            new_list.append(item)
                    final_value = new_list
                elif isinstance(current_value, str):
                    # 既存値が文字列の場合は、リストに変換して先頭に new_value を追加
                    final_value = [new_value, current_value] if current_value != new_value else [new_value]
                else:
                    # 既存値がない場合は、new_value のみのリスト
                    final_value = [new_value]
                
                diff_ops.append({
                    "op": "replace",
                    "path": json_path,
                    "value": final_value,
                    "metadata": {
                        "action": "selector_patch",
                        "old_value": old_value,
                        "confidence": confidence,
                    },
                })
                
            else:
                logger.warning(f"[PatchAdapter] Unknown action: {action} for path: {path}")
        
        return diff_ops
    
    def _to_json_pointer(self, dot_path: str) -> str:
        """
        dot 区切りのパスを JSON Pointer 形式に変換
        
        例:
        - "discovery_settings.timeout_sec" -> "/discovery_settings/timeout_sec"
        - "selectors.pdp" -> "/selectors/pdp"
        """
        parts = dot_path.split(".")
        return "/" + "/".join(parts)
    
    def _get_nested_value(self, obj: Dict[str, Any], dot_path: str, default: Any = None) -> Any:
        """
        dot 区切りのパスでネストされた値を取得
        
        例:
        - "discovery_settings.timeout_sec" -> obj["discovery_settings"]["timeout_sec"]
        """
        parts = dot_path.split(".")
        current = obj
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        
        return current
    
    def _path_exists(self, obj: Dict[str, Any], dot_path: str) -> bool:
        """
        dot 区切りのパスが存在するかチェック
        """
        parts = dot_path.split(".")
        current = obj
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return False
        
        return True

