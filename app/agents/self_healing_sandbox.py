# -*- coding: utf-8 -*-
"""
SelfHealingSandbox - Self-Healing パッチ適用の仮想環境（Sandbox）

CR-ATELIER-003 Phase D-8: メモリ上でパッチを適用し、本番ファイルを変更せずに
Self-Healing Loop を試運転できる環境を提供する。
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional

from app.agents.self_healing_patch_adapter import SelfHealingPatchAdapter
from app.agents.self_healing_patch_applier import SelfHealingPatchApplier

logger = logging.getLogger(__name__)


class SelfHealingSandbox:
    """
    Self-Healing パッチ適用の仮想環境（Sandbox）
    
    overrides.local.json をメモリ上でコピーし、パッチを適用した結果を返す。
    本番ファイルは一切変更しない。
    """
    
    def __init__(
        self,
        *,
        patch_adapter: Optional[SelfHealingPatchAdapter] = None,
        patch_applier: Optional[SelfHealingPatchApplier] = None,
    ) -> None:
        """
        SelfHealingSandbox を初期化
        
        Args:
            patch_adapter: SelfHealingPatchAdapter インスタンス（オプション）
            patch_applier: SelfHealingPatchApplier インスタンス（オプション）
        """
        self.adapter = patch_adapter or SelfHealingPatchAdapter()
        self.applier = patch_applier or SelfHealingPatchApplier(patch_adapter=self.adapter)
        self.logger = logger
        # CR-ATELIER-003 Phase D-10.1: セレクタ成功/失敗ログを保存
        self._selector_results: Dict[str, List[Dict[str, Any]]] = {}  # key: "{site}:{page_type}"
    
    def apply_patch_in_memory(
        self,
        *,
        overrides_json: Dict[str, Any],
        patch_candidate: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        overrides_json を破壊せず、パッチ適用後の仮想 overrides を返す
        
        Args:
            overrides_json: 現行の overrides.local.json の内容（辞書形式）
            patch_candidate: Phase D-6 で生成された patch_candidate 辞書
        
        Returns:
            パッチ適用後の仮想 overrides 辞書。元の overrides_json は変更されない。
        """
        # ディープコピーを作成（元の辞書を変更しない）
        virtual_overrides = copy.deepcopy(overrides_json)
        
        target_site = patch_candidate.get("target_site", "unknown")
        
        # 対象サイトが存在しない場合
        if target_site not in virtual_overrides:
            self.logger.warning(
                f"[Sandbox] Site '{target_site}' not found in overrides_json. "
                "Returning original overrides without changes."
            )
            return virtual_overrides
        
        site_config = virtual_overrides[target_site]
        
        # パッチ候補を diff 形式に変換
        diff_ops = self.adapter.to_diff(
            patch_candidate=patch_candidate,
            site_config=site_config,
        )
        
        if not diff_ops:
            self.logger.info(f"[Sandbox] No changes to apply for {target_site}")
            return virtual_overrides
        
        # diff を適用（メモリ上のコピーに対して）
        try:
            self._apply_diff_ops_in_memory(
                overrides=virtual_overrides,
                target_site=target_site,
                diff_ops=diff_ops,
            )
            
            self.logger.info(
                f"[Sandbox] Patch applied successfully in memory: "
                f"target_site={target_site}, diff_ops_count={len(diff_ops)}"
            )
            
        except Exception as e:
            self.logger.error(
                f"[Sandbox] Failed to apply diff in memory: {e}",
                exc_info=True
            )
            # エラーが発生した場合、元の overrides を返す
            return copy.deepcopy(overrides_json)
        
        return virtual_overrides
    
    def get_site_config_from_overrides(
        self,
        *,
        overrides_json: Dict[str, Any],
        site_code: str,
    ) -> Optional[Dict[str, Any]]:
        """
        仮想 overrides から特定サイトの site_config を取得
        
        Args:
            overrides_json: overrides 辞書（apply_patch_in_memory の結果など）
            site_code: サイトコード
        
        Returns:
            サイト設定辞書。存在しない場合は None。
        """
        return overrides_json.get(site_code)
    
    def _apply_diff_ops_in_memory(
        self,
        overrides: Dict[str, Any],
        target_site: str,
        diff_ops: list[Dict[str, Any]],
    ) -> None:
        """
        diff 操作を overrides の target_site ブロックに適用する（メモリ上）
        
        Args:
            overrides: overrides.local.json の内容（変更可能なコピー）
            target_site: 対象サイトコード
            diff_ops: 適用する diff 操作のリスト
        """
        site_config = overrides[target_site]
        
        for op in diff_ops:
            op_type = op.get("op")
            path = op.get("path")
            value = op.get("value")
            
            if not op_type or not path:
                continue
            
            # JSON Pointer から dot 区切りのパスに変換（site_config 内の相対パス）
            # 例: "/discovery_settings/timeout_sec" -> "discovery_settings.timeout_sec"
            if path.startswith("/"):
                path = path[1:]  # 先頭の "/" を削除
            
            dot_path = path.replace("/", ".")
            parts = dot_path.split(".")
            
            # ネストされた値を設定
            current = site_config
            for i, part in enumerate(parts[:-1]):
                if part not in current:
                    current[part] = {}
                elif not isinstance(current[part], dict):
                    # 既存の値が dict でない場合は dict に置き換え
                    current[part] = {}
                current = current[part]
            
            final_key = parts[-1]
            
            if op_type == "add":
                current[final_key] = value
            elif op_type == "replace":
                if final_key not in current:
                    raise ValueError(f"Path '{dot_path}' does not exist for replace operation")
                current[final_key] = value
            elif op_type == "remove":
                if final_key in current:
                    del current[final_key]
            else:
                raise ValueError(f"Unknown operation: {op_type}")
    
    def record_selector_result(
        self,
        *,
        site: str,
        selector: str,
        success: bool,
        kind: str = "pdp_link",
        reason: Optional[str] = None,
        page_type: Optional[str] = None,  # 後方互換性のため保持
        selector_result: Optional[Dict[str, Any]] = None,  # 後方互換性のため保持
    ) -> None:
        """
        CR-ATELIER-003 Phase D-10.1 Integration: セレクタの検証結果を記録する
        
        Args:
            site: サイトコード
            selector: CSS セレクタ文字列
            success: 成功したかどうか
            kind: セレクタの種類（"pdp_link", "price", "image" など、デフォルト: "pdp_link"）
            reason: 成功/失敗の理由（オプション）
            page_type: ページタイプ（"plp" または "pdp"、後方互換性のため、kind から推論可能）
            selector_result: セレクタ検証結果辞書（後方互換性のため、個別パラメータ優先）
        """
        # 後方互換性: selector_result が渡された場合は個別パラメータに変換
        if selector_result:
            selector = selector_result.get("selector", selector)
            success = selector_result.get("success", success)
            reason = selector_result.get("reason", reason)
        
        # page_type が指定されていない場合、kind から推論
        if not page_type:
            if kind == "pdp_link":
                page_type = "plp"  # PLP から PDP リンクを抽出
            else:
                page_type = "pdp"  # デフォルト
        
        key = f"{site}:{page_type}"
        if key not in self._selector_results:
            self._selector_results[key] = []
        
        result_entry = {
            "selector": selector,
            "success": success,
            "kind": kind,
            "reason": reason or ("Success" if success else "Failed"),
        }
        
        self._selector_results[key].append(result_entry)
        
        # 最大100件まで保持（古いものから削除）
        if len(self._selector_results[key]) > 100:
            self._selector_results[key] = self._selector_results[key][-100:]
    
    def get_previous_successes(
        self,
        *,
        site: str,
        kind: str = "pdp_link",
        page_type: Optional[str] = None,  # 後方互換性のため保持
    ) -> List[Dict[str, Any]]:
        """
        CR-ATELIER-003 Phase D-10.1 Integration: 過去の成功したセレクタを取得
        
        Args:
            site: サイトコード
            kind: セレクタの種類（"pdp_link", "price", "image" など、デフォルト: "pdp_link"）
            page_type: ページタイプ（"plp" または "pdp"、後方互換性のため、kind から推論可能）
        
        Returns:
            成功したセレクタのリスト（各要素は {"selector": str, "success": bool, "kind": str, "reason": str}）
        """
        # page_type が指定されていない場合、kind から推論
        if not page_type:
            if kind == "pdp_link":
                page_type = "plp"
            else:
                page_type = "pdp"
        
        key = f"{site}:{page_type}"
        results = self._selector_results.get(key, [])
        # kind でフィルタリング（指定された場合）
        filtered = [r for r in results if r.get("success", False)]
        if kind != "pdp_link":  # デフォルト以外の場合は kind でフィルタ
            filtered = [r for r in filtered if r.get("kind") == kind]
        return filtered
    
    def get_previous_failures(
        self,
        *,
        site: str,
        kind: str = "pdp_link",
        page_type: Optional[str] = None,  # 後方互換性のため保持
    ) -> List[Dict[str, Any]]:
        """
        CR-ATELIER-003 Phase D-10.1 Integration: 過去の失敗したセレクタを取得
        
        Args:
            site: サイトコード
            kind: セレクタの種類（"pdp_link", "price", "image" など、デフォルト: "pdp_link"）
            page_type: ページタイプ（"plp" または "pdp"、後方互換性のため、kind から推論可能）
        
        Returns:
            失敗したセレクタのリスト（各要素は {"selector": str, "success": bool, "kind": str, "reason": str}）
        """
        # page_type が指定されていない場合、kind から推論
        if not page_type:
            if kind == "pdp_link":
                page_type = "plp"
            else:
                page_type = "pdp"
        
        key = f"{site}:{page_type}"
        results = self._selector_results.get(key, [])
        # kind でフィルタリング（指定された場合）
        filtered = [r for r in results if not r.get("success", True)]
        if kind != "pdp_link":  # デフォルト以外の場合は kind でフィルタ
            filtered = [r for r in filtered if r.get("kind") == kind]
        return filtered

