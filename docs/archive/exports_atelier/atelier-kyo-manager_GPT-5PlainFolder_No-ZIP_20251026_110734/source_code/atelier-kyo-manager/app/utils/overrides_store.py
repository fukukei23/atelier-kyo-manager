# ==============================================================================
# ファイル名 (File Name): app/utils/overrides_store.py
# レジストリ (Registry): app/utils/overrides_store.py
# 更新日時 (Date & Time JST): 2025年10月03日 00:35:00
# バージョン (Version): 9.0.0J (Production-Grade: Failsafe & Compatible)
#
# --- v9.0.0Jでの主な変更点 ---
# - [フェイルセーフI/O] あなたの提案に基づき、ファイル書き込み前に親ディレクトリの
#   存在を保証する処理を追加。クリーンな環境での初回実行時のエラーを防止します。
# - [API互換性の復活] 外部からセレクタを安全に更新するための便利なAPI
#   `update_site_selectors`を、v8の全ての安全機構（アトミック書き込み、
#   安全マージ、監査ログ等）を完全に継承した形で復活させました。
#
# --- 既存機能からの維持 ---
# - v8.0.0Jの無停止バックアップ、自己修復ファイルシステム、アトミックな書き込み、
#   監査性、データ検証といった全ての堅牢化機能は完全に維持されています。
# ==============================================================================
# -*- coding: utf-8 -*-

from __future__ import annotations
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
import difflib

logger = logging.getLogger(__name__)

# ==========================
# 定数
# ==========================
BACKUP_DIR = Path(".overrides_backups")
MAX_BACKUPS = 50

# ==========================
# 内部ユーティリティ (v8.0.0Jから変更なし)
# ==========================
def _is_valid_selector_list_item(item: Any) -> bool:
    return isinstance(item, (str, dict))

def _sanitize_selector_list(items: List[Any]) -> List[Any]:
    return [item for item in items if _is_valid_selector_list_item(item)]

def _unique_extend(base: List[Any], new: List[Any]) -> List[Any]:
    seen = set()
    out: List[Any] = []
    for x in new + base:
        key = tuple(sorted(x.items())) if isinstance(x, dict) else x
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out

def _strip_empty_lists(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            vv = _strip_empty_lists(v)
            if vv: out[k] = vv
        elif isinstance(v, list):
            sanitized_v = _sanitize_selector_list(v)
            if sanitized_v: out[k] = sanitized_v
        else:
            out[k] = v
    return out

def _deep_merge_safe(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge_safe(out[k], v)
        elif k in out and isinstance(out[k], list) and isinstance(v, list):
            sanitized_v = _sanitize_selector_list(v)
            if sanitized_v:
                out[k] = _unique_extend(out[k], sanitized_v)
        else:
            out[k] = v
    return out

def _rotate_backups(basename: str):
    try:
        backups = sorted(
            BACKUP_DIR.glob(f"{basename}.*.bak"),
            key=os.path.getmtime,
            reverse=True
        )
        if len(backups) > MAX_BACKUPS:
            for old_backup in backups[MAX_BACKUPS:]:
                old_backup.unlink()
                logger.debug(f"Rotated old backup: {old_backup}")
    except Exception as e:
        logger.warning(f"Failed to rotate backups: {e}")

def _backup_overrides_file(path: Path, reason: str = "update") -> None:
    try:
        if not path.exists(): return
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        bak_name = f"{path.name}.{reason}.{ts}.bak"
        bak = BACKUP_DIR / bak_name

        if reason == "corrupted":
            os.rename(path, bak)
            logger.info(f"Isolated corrupted '{path.name}' to '{bak_name}'.")
        else:
            shutil.copy2(path, bak)
            logger.info(f"Backed up '{path.name}' to '{bak_name}' (reason: {reason}).")

        _rotate_backups(path.name)
    except Exception as e:
        logger.warning(f"バックアップ作成に失敗しました: {e}")

def _as_selectors_dict(body: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(body, dict): return {}
    if "selectors" in body and isinstance(body["selectors"], dict):
        return {"selectors": body["selectors"]}
    if "suggested_selectors" in body and isinstance(body["suggested_selectors"], dict):
        return {"selectors": body["suggested_selectors"]}
    return {"selectors": body}

def _normalize_proposal(proposal: Dict[str, Any], site: Optional[str] = None) -> Dict[str, Any]:
    if not isinstance(proposal, dict) or not proposal: return {}
    if site:
        if site in proposal and isinstance(proposal[site], dict):
            return {site: _as_selectors_dict(proposal[site])}
        return {site: _as_selectors_dict(proposal)}
    top_keys = list(proposal.keys())
    if len(top_keys) == 1 and isinstance(proposal[top_keys[0]], dict):
        inferred_site = top_keys[0]
        return {inferred_site: _as_selectors_dict(proposal[inferred_site])}
    raise ValueError(
        "normalize_proposal: 'site' を指定するか、proposal を {<SITE_NAME>: {...}} 形式にしてください。"
    )

# ==========================
# 公開API
# ==========================
def extract_suggested_selectors(
    evidence: Optional[Dict[str, Any]],
    target: str = "pdp.pdp_link_selectors",
    defaults: Optional[List[str]] = None
) -> Optional[List[str]]:
    if not evidence or not isinstance(evidence, dict): return defaults
    selectors_root = evidence.get("suggested_selectors")
    if not isinstance(selectors_root, dict): return defaults
    current = selectors_root
    keys = target.split('.')
    for key in keys:
        if not isinstance(current, dict) or key not in current: return defaults
        current = current[key]
    if isinstance(current, list):
        return _sanitize_selector_list(current)
    return defaults

def apply_selector_proposal(
    *,
    proposal_path: Union[str, Path],
    overrides_path: Union[str, Path],
    site: str
) -> Tuple[bool, Optional[str]]:
    proposal_p = Path(proposal_path)
    overrides_file = Path(overrides_path)
    try:
        if not proposal_p.exists():
            logger.warning(f"提案ファイルが見つかりません: {proposal_p}")
            return False, None

        proposal_data = json.loads(proposal_p.read_text(encoding="utf-8"))
        normalized = _normalize_proposal(proposal_data, site=site)

        site_payload = _strip_empty_lists(normalized.get(site, {}))
        if not site_payload.get("selectors"):
            logger.info("適用可能な差分が見つかりませんでした。スキップします。")
            return False, None

        before_content = "{}"
        overrides_data = {}
        if overrides_file.exists():
            try:
                before_content = overrides_file.read_text(encoding="utf-8")
                overrides_data = json.loads(before_content)
            except (json.JSONDecodeError, OSError):
                logger.error(f"'{overrides_file}'が破損。隔離バックアップして空から再生成します。", exc_info=True)
                _backup_overrides_file(overrides_file, reason="corrupted")
                before_content = "{}"
                overrides_data = {}

        merged = _deep_merge_safe(overrides_data, {site: site_payload})
        after_content = json.dumps(merged, indent=2, ensure_ascii=False)

        if before_content == after_content:
            logger.info("差分がなかったため、ファイル更新をスキップしました。")
            return False, None

        _backup_overrides_file(overrides_file, reason="pre-update")

        # ▼▼▼ v9.0.0J 修正点 ▼▼▼
        overrides_file.parent.mkdir(parents=True, exist_ok=True)
        # ▲▲▲ v9.0.0J 修正点 ▲▲▲

        tmp_file = overrides_file.with_suffix(f"{overrides_file.suffix}.tmp")
        tmp_file.write_text(after_content, encoding="utf-8")
        tmp_file.replace(overrides_file)

        logger.info(f"サイト '{site}' のセレクタ提案を {overrides_file} に安全にマージしました。")

        diff = "".join(difflib.unified_diff(
            before_content.splitlines(keepends=True),
            after_content.splitlines(keepends=True),
            fromfile=f"before: {overrides_file.name}", tofile=f"after: {overrides_file.name}"
        ))
        return True, diff

    except Exception as e:
        logger.error(f"セレクタ提案のマージ処理中に予期せぬエラー: {e}", exc_info=True)
        return False, None

# ▼▼▼ v9.0.0J 修正点 ▼▼▼
def update_site_selectors(
    *,
    site: str,
    new_selectors: Dict[str, Any],
    overrides_path: Union[str, Path]
) -> Tuple[bool, Optional[str]]:
    """
    既存のoverridesに対し、{site: {selectors: new_selectors}} を安全マージ。
    """
    overrides_file = Path(overrides_path)
    new_payload = _strip_empty_lists({"selectors": new_selectors})
    if not new_payload.get("selectors"):
        logger.info("update_site_selectors: 適用可能な差分なし。スキップします。")
        return False, None

    before_content = "{}"
    overrides_data: Dict[str, Any] = {}
    if overrides_file.exists():
        try:
            before_content = overrides_file.read_text(encoding="utf-8")
            overrides_data = json.loads(before_content)
        except (json.JSONDecodeError, OSError):
            logger.error(f"'{overrides_file}'が破損。隔離バックアップして空から再生成します。", exc_info=True)
            _backup_overrides_file(overrides_file, reason="corrupted")
            before_content = "{}"
            overrides_data = {}

    merged = _deep_merge_safe(overrides_data, {site: new_payload})
    after_content = json.dumps(merged, indent=2, ensure_ascii=False)

    if before_content == after_content:
        logger.info("update_site_selectors: 差分が無いため更新をスキップしました。")
        return False, None

    _backup_overrides_file(overrides_file, reason="pre-update")

    overrides_file.parent.mkdir(parents=True, exist_ok=True)

    tmp_file = overrides_file.with_suffix(f"{overrides_file.suffix}.tmp")
    tmp_file.write_text(after_content, encoding="utf-8")
    tmp_file.replace(overrides_file)

    diff = "".join(difflib.unified_diff(
        before_content.splitlines(keepends=True),
        after_content.splitlines(keepends=True),
        fromfile=f"before: {overrides_file.name}",
        tofile=f"after:  {overrides_file.name}",
    ))
    logger.info(f"update_site_selectors: サイト '{site}' を更新しました。")
    return True, diff
# ▲▲▲ v9.0.0J 修正点 ▲▲▲

