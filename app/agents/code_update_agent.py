# ==============================================================================
# File: app/agents/code_update_agent.py
# Date (JST): 2025-11-05 13:02:00
# Version: 1.3.2-apply-orchestrator
#
# 目的:
# - v1.3.1 の機能（apply_overrides_patch, write_unified_diff）を維持。
# - [v1.3.2] ★統合: diff の意図を汲み、高レベルの apply メソッドを追加。
# - apply は proposal を受け取り、簡易監査ログ(selectors/overrides)を
#   exports/patches に保存した後、既存の apply_overrides_patch を呼び出す。
# ==============================================================================
import contextlib
import difflib
import json
import logging
import os  # ★ NEW (diff で要求)
import subprocess
import time
from datetime import datetime  # ★ NEW (diff で要求)
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CodeUpdateAgent:
    """
    - [v1.3.2] apply: 提案全体を受け取り、簡易監査ログを保存し、下位メソッドを呼ぶ。
    - apply_overrides_patch: overrides.local.json に最小差分を適用（dry_run可）
    - before/after の差分スナップショットを instance/patches/ に保存
    - [v1.3.1] unified diff を instance/patches/ に保存
    - enable_git_stage=True で、生成物を git add
    """

    def __init__(
        self,
        overrides_path: str = "app/config/sites/overrides.local.json",
        patch_dir: str = "instance/patches",
        enable_git_stage: bool = False,
    ):
        self.overrides_path = Path(overrides_path)
        self.patch_dir = Path(patch_dir)
        self.patch_dir.mkdir(parents=True, exist_ok=True)
        self.enable_git_stage = enable_git_stage

    # ------------------------------------------------------------------
    # ★統合(v1.3.2): diff に基づく高レベルの apply メソッド
    # ------------------------------------------------------------------
    def apply(self, proposal: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
        """
        AI提案(proposal)全体を受け取り、監査ログを保存し、
        適用可能なパッチ（現在はoverrides）を適用するオーケストレーター。

        proposal: { selectors_patch, overrides_patch, code_hints, site? ... }
        """
        out = {"applied": False, "notes": []}

        # --- ALWAYS dump diffs/patch-like text for auditing (diff の意図) ---
        try:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            site = (proposal.get("site") or proposal.get("site_key") or "unknown").lower()
            patches_dir = os.path.join("exports", "patches")
            os.makedirs(patches_dir, exist_ok=True)

            # selectors / overrides を疑似 unified diff で保存（監査用）
            sel_patch = proposal.get("selectors_patch") or {}
            ovr_patch = proposal.get("overrides_patch") or {}

            if sel_patch:
                sel_path = os.path.join(patches_dir, f"{ts}_{site}_selectors.diff")
                with open(sel_path, "w", encoding="utf-8") as f:
                    f.write("--- a/selectors.json\n+++ b/selectors.json\n")
                    for k, v in sel_patch.items():
                        f.write(f"+ {k}: {v}\n")
                out["notes"].append(f"Wrote selectors audit patch: {sel_path}")

            if ovr_patch:
                ovr_path = os.path.join(patches_dir, f"{ts}_{site}_overrides.diff")
                with open(ovr_path, "w", encoding="utf-8") as f:
                    f.write("--- a/overrides.local.json\n+++ b/overrides.local.json\n")
                    f.write("+ patch (merged keys):\n")
                    for k in sorted(ovr_patch.keys()):
                        f.write(f"+    {k}: {ovr_patch[k]}\n")
                out["notes"].append(f"Wrote overrides audit patch: {ovr_path}")

            logger.info(f"[CodeUpdateAgent.apply] wrote audit patches → {patches_dir}")
        except Exception as e:
            logger.warning(f"[CodeUpdateAgent.apply] failed to write audit patches: {e}")
            out["notes"].append(f"Failed to write audit patches: {e}")

        # dry-run の場合は適用しない (diff の意図)
        if dry_run:
            out["notes"].append("dry_run: patches dumped only")
            return out

        # --- "existing logic" (v1.3.1) を呼び出し ---
        # (現在は overrides のみ適用)
        ovr_patch = proposal.get("overrides_patch") or {}
        site_key = (proposal.get("site") or proposal.get("site_key") or "unknown").lower()

        if ovr_patch:
            try:
                # 既存の堅牢な適用メソッドを呼び出す
                applied_path = self.apply_overrides_patch(
                    site_key=site_key,
                    overrides_patch=ovr_patch,
                    dry_run=dry_run,  # dry_run=False だが、念のため渡す
                )
                if applied_path:
                    out["notes"].append(f"Overrides patch applied (see jsondiff): {applied_path}")
                    out["applied"] = True  # 実際に適用された場合のみ True
                else:
                    out["notes"].append("Overrides patch not applied (no changes detected).")
            except Exception as e:
                logger.warning(f"[CodeUpdateAgent.apply] apply_overrides_patch failed: {e}")
                out["notes"].append(f"Overrides patch application failed: {e}")
        else:
            out["notes"].append("No overrides_patch found in proposal, skipping application.")

        # Note: selectors_patch は監査されますが、適用ロジックはこのクラスにありません。
        # code_hints も同様です。

        return out

    # ------------------------------------------------------------------
    # 既存の v1.3.1 メソッド
    # ------------------------------------------------------------------
    def apply_overrides_patch(
        self, site_key: str, overrides_patch: dict[str, Any], *, dry_run: bool = False
    ) -> Path | None:
        """
        [v1.3.1] 堅牢な overrides.local.json 適用メソッド
        """
        if not overrides_patch:
            return None

        before = self._read_json(self.overrides_path) or {}
        merged = self._deep_merge_site(before, site_key, overrides_patch)

        if merged == before:
            logger.info("[CodeUpdate] overrides に変更なし（スキップ）")
            return None

        # 競合/危険変更の簡易検出
        ok, reason = self._validate_merge(before.get(site_key, {}), merged.get(site_key, {}))
        if not ok:
            logger.warning(f"[CodeUpdate] 競合疑いのため適用を中止: {reason}")
            return self._dump_conflict(site_key, before, merged, reason)

        # diff（before/after）を保存 (v1.3.1 の資産)
        stamp = self._stamp()
        diff_path = self.patch_dir / f"overrides_patch_{site_key}_{stamp}.jsondiff"
        diff_path.write_text(
            self._json({"before": before.get(site_key, {}), "after": merged.get(site_key, {})}), encoding="utf-8"
        )

        if dry_run:
            logger.info("[CodeUpdate] dry_run のためファイル更新は行いません (jsondiff のみ保存)")
            return diff_path

        # バックアップ
        self._backup_json(self.overrides_path)
        # 実ファイル更新
        self._write_json(self.overrides_path, merged)
        logger.info(f"[CodeUpdate] overrides を更新: {self.overrides_path}")

        if self.enable_git_stage:
            self._git_add(self.overrides_path)
            self._git_add(diff_path)

        return diff_path

    def emit_code_diff(self, title: str, diff_text: str) -> Path:
        stamp = self._stamp()
        path = self.patch_dir / f"{title}_{stamp}.diff"
        path.write_text(diff_text, encoding="utf-8")
        logger.info(f"[CodeUpdate] diff を出力: {path}")
        if self.enable_git_stage:
            self._git_add(path)
        return path

    # --- ★追加(v1.3.1): unified diff をファイル出力 ---
    def write_unified_diff(
        self, old_text: str, new_text: str, target_path: str, base_dir: str | None = None
    ) -> str | None:
        """
        2つのテキストから unified diff を生成し、.diff ファイルとして保存する。
        （既存の self.patch_dir ではなく、diff提案のロジックに基づき
         PROXY_OUTPUT_DIR or base_dir を優先）
        """
        try:
            # 出力先ディレクトリの決定 (diff提案ロジックを尊重)
            out_dir_root = Path(base_dir) if base_dir else Path(os.environ.get("PROXY_OUTPUT_DIR", "exports"))

            out_dir = out_dir_root / "patches"
            out_dir.mkdir(parents=True, exist_ok=True)

            ts = time.strftime("%Y%m%d_%H%M%S")
            # target_path からファイル名のみを取得
            fname = f"{Path(target_path).name}.{ts}.diff"
            out_path = out_dir / fname

            # unified diff の生成
            diff = difflib.unified_diff(
                old_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=f"a/{target_path}",
                tofile=f"b/{target_path}",
            )

            # ファイルへの書き込み
            with open(out_path, "w", encoding="utf-8") as f:
                f.writelines(diff)

            logger.info(f"[CodeUpdateAgent] wrote unified diff: {out_path}")

            # ★ 統合: 既存のGit連携機能を活用
            if self.enable_git_stage:
                self._git_add(out_path)

            return str(out_path)  # Path を str にして返す

        except Exception as e:
            logger.warning(f"[CodeUpdateAgent] failed to write diff: {e}")
            return None

    # ------------------------------------------------------------------
    # Helpers (v1.3.1)
    # ------------------------------------------------------------------
    def _read_json(self, p: Path) -> dict[str, Any]:
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8")) or {}
        except Exception as e:
            logger.warning(f"[CodeUpdate] JSON read失敗: {p} - {e}")
            return {}

    def _write_json(self, p: Path, data: dict[str, Any]) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self._json(data), encoding="utf-8")

    def _backup_json(self, p: Path) -> Path | None:
        if not p.exists():
            return None
        bak = p.with_suffix(p.suffix + f".bak.{self._stamp()}")
        bak.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
        logger.info(f"[CodeUpdate] backup: {bak}")
        return bak

    def _deep_merge_site(self, all_data: dict[str, Any], site_key: str, patch: dict[str, Any]) -> dict[str, Any]:
        data = dict(all_data)
        base = data.get(site_key, {})
        data[site_key] = self._deep_merge(base, patch)
        return data

    def _deep_merge(self, a: Any, b: Any) -> Any:
        if isinstance(a, dict) and isinstance(b, dict):
            out = dict(a)
            for k, v in b.items():
                out[k] = self._deep_merge(a.get(k), v) if k in a else v
            return out
        return b

    def _validate_merge(self, before: dict[str, Any], after: dict[str, Any]) -> tuple[bool, str]:
        # 型崩れを検出（dict→list等）
        if not isinstance(before, dict) or not isinstance(after, dict):
            return False, "non-dict merge detected"
        # 大量削除（キー数が半減以下）は危険度高
        if len(after.keys()) < max(1, len(before.keys()) // 2):
            return False, "suspicious mass deletion"
        return True, "ok"

    def _dump_conflict(self, site_key: str, before: dict[str, Any], merged: dict[str, Any], reason: str) -> Path:
        path = self.patch_dir / f"overrides_conflict_{site_key}_{self._stamp()}.json"
        path.write_text(
            self._json(
                {
                    "reason": reason,
                    "before": before.get(site_key, {}),
                    "merged": merged.get(site_key, {}),
                }
            ),
            encoding="utf-8",
        )
        logger.info(f"[CodeUpdate] 競合スナップショット: {path}")
        return path

    def _json(self, data: dict[str, Any]) -> str:
        return json.dumps(data, ensure_ascii=False, indent=2)

    def _stamp(self) -> str:
        return datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    def _git_add(self, path: Path) -> None:
        with contextlib.suppress(Exception):
            subprocess.run(["git", "add", str(path)], check=False, capture_output=True)
