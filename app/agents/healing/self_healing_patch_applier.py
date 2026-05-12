"""
SelfHealingPatchApplier - Self-Healing パッチ候補を overrides.local.json に適用する

CR-ATELIER-003 Phase D-7: patch_candidate_self_healing.json を読み込み、
overrides.local.json に安全に適用する。
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from app.agents.healing.self_healing_patch_adapter import SelfHealingPatchAdapter

logger = logging.getLogger(__name__)


class SelfHealingPatchApplier:
    """
    Self-Healing パッチ候補を overrides.local.json に適用する

    Phase D-6 で生成された patch_candidate_self_healing.json を読み込み、
    overrides.local.json に安全に適用する。
    """

    def __init__(self, *, patch_adapter: SelfHealingPatchAdapter | None = None) -> None:
        """
        SelfHealingPatchApplier を初期化

        Args:
            patch_adapter: SelfHealingPatchAdapter インスタンス（オプション）
        """
        self.adapter = patch_adapter or SelfHealingPatchAdapter()
        self.logger = logger

    def apply_patch_candidate(
        self,
        *,
        candidate_path: Path,
        overrides_path: Path,
        backup_suffix: str | None = None,
    ) -> dict[str, Any]:
        """
        パッチ候補を overrides.local.json に適用する

        Args:
            candidate_path: patch_candidate_self_healing.json のパス
            overrides_path: overrides.local.json のパス
            backup_suffix: バックアップファイルの suffix（例: ".bak-20251210-120000"）
                           None の場合は自動生成

        Returns:
            適用結果の辞書。以下のフィールドを含む:
                - applied: bool（適用が成功したか）
                - diff_ops: List[Dict]（適用された diff 操作）
                - backup_path: Path（バックアップファイルのパス）
                - target_site: str（対象サイトコード）
                - error: Optional[str]（エラーメッセージ、失敗時のみ）
        """
        try:
            # パッチ候補を読み込む
            if not candidate_path.exists():
                return {
                    "applied": False,
                    "diff_ops": [],
                    "backup_path": None,
                    "target_site": None,
                    "error": f"Patch candidate file not found: {candidate_path}",
                }

            with open(candidate_path, encoding="utf-8") as f:
                patch_candidate = json.load(f)

            target_site = patch_candidate.get("target_site", "unknown")

            # overrides.local.json を読み込む
            if not overrides_path.exists():
                return {
                    "applied": False,
                    "diff_ops": [],
                    "backup_path": None,
                    "target_site": target_site,
                    "error": f"Overrides file not found: {overrides_path}",
                }

            with open(overrides_path, encoding="utf-8") as f:
                overrides = json.load(f)

            # 対象サイトの site_config を取得
            if target_site not in overrides:
                return {
                    "applied": False,
                    "diff_ops": [],
                    "backup_path": None,
                    "target_site": target_site,
                    "error": f"Site '{target_site}' not found in overrides",
                }

            site_config = overrides[target_site]

            # パッチ候補を diff 形式に変換
            diff_ops = self.adapter.to_diff(
                patch_candidate=patch_candidate,
                site_config=site_config,
            )

            if not diff_ops:
                self.logger.info(f"[PatchApplier] No changes to apply for {target_site}")
                return {
                    "applied": False,
                    "diff_ops": [],
                    "backup_path": None,
                    "target_site": target_site,
                    "error": "No changes to apply",
                }

            # バックアップを作成
            if backup_suffix is None:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                backup_suffix = f".bak-{timestamp}"

            backup_path = overrides_path.with_suffix(overrides_path.suffix + backup_suffix)
            shutil.copy2(overrides_path, backup_path)
            self.logger.info(f"[PatchApplier] Backup created: {backup_path}")

            # diff を適用
            try:
                self._apply_diff_ops(overrides, target_site, diff_ops)
            except Exception as e:
                # 適用に失敗した場合、バックアップから復元
                self.logger.error(f"[PatchApplier] Failed to apply diff, restoring from backup: {e}", exc_info=True)
                shutil.copy2(backup_path, overrides_path)
                return {
                    "applied": False,
                    "diff_ops": diff_ops,
                    "backup_path": backup_path,
                    "target_site": target_site,
                    "error": f"Failed to apply diff: {str(e)}",
                }

            # 更新された overrides を保存
            with open(overrides_path, "w", encoding="utf-8") as f:
                json.dump(overrides, f, indent=2, ensure_ascii=False)

            self.logger.info(
                f"[PatchApplier] Patch applied successfully: "
                f"target_site={target_site}, diff_ops_count={len(diff_ops)}, backup={backup_path}"
            )

            return {
                "applied": True,
                "diff_ops": diff_ops,
                "backup_path": backup_path,
                "target_site": target_site,
                "error": None,
            }

        except json.JSONDecodeError as e:
            return {
                "applied": False,
                "diff_ops": [],
                "backup_path": None,
                "target_site": None,
                "error": f"Invalid JSON: {str(e)}",
            }
        except Exception as e:
            self.logger.error(f"[PatchApplier] Unexpected error: {e}", exc_info=True)
            return {
                "applied": False,
                "diff_ops": [],
                "backup_path": None,
                "target_site": None,
                "error": f"Unexpected error: {str(e)}",
            }

    def _apply_diff_ops(
        self,
        overrides: dict[str, Any],
        target_site: str,
        diff_ops: List[dict[str, Any]],
    ) -> None:
        """
        diff 操作を overrides の target_site ブロックに適用する

        Args:
            overrides: overrides.local.json の内容
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
            for _i, part in enumerate(parts[:-1]):
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
