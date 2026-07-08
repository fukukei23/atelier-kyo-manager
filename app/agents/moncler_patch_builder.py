from __future__ import annotations

import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypedDict

logger: logging.Logger = logging.getLogger(__name__)


# ── Types ────────────────────────────────────────────────


class RiskLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class RiskAssessment(TypedDict):
    overall: str
    notes: str


class SelectorChange(TypedDict, total=False):
    before: list[str]
    after: list[str]


class SelectorLayerChange(TypedDict, total=False):
    pdp_link_selectors: SelectorChange
    tile_selectors: SelectorChange


class TrapAppend(TypedDict):
    append: list[str]


# ── Payload builder ──────────────────────────────────────


def build_moncler_analysis_payload(
    run_id: str,
    site: str,
    current_url: str,
    moncler_outcome: dict[str, Any],
    self_healing_result: dict[str, Any] | None = None,
    selector_discovery_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": "1.0",
        "run_id": run_id,
        "site": site,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "current_url": current_url,
        "moncler_outcome": moncler_outcome,
    }

    if self_healing_result:
        suggested_actions: list[dict[str, Any]] = []
        for action in self_healing_result.get("suggested_actions", []):
            if isinstance(action, str):
                suggested_actions.append(
                    {"id": f"action_{len(suggested_actions)}", "description": action, "risk_level": "MEDIUM"}
                )
            elif isinstance(action, dict):
                suggested_actions.append(action)
        payload["self_healing"] = {
            "analysis": self_healing_result.get("analysis", ""),
            "root_cause": self_healing_result.get("root_cause", ""),
            "suggested_actions": suggested_actions,
            "confidence": self_healing_result.get("confidence", 0.0),
        }

    if selector_discovery_result:
        candidates = selector_discovery_result.get("candidate_selectors", [])
        scores = selector_discovery_result.get("confidence_scores", [])
        confidence_dict: dict[str, float] = {}
        if isinstance(scores, list):
            for i, sel in enumerate(candidates):
                if i < len(scores):
                    confidence_dict[sel] = scores[i]
        elif isinstance(scores, dict):
            confidence_dict = scores
        payload["selector_discovery"] = {
            "candidate_selectors": candidates,
            "recommended_layer": selector_discovery_result.get("recommended_layer", "primary"),
            "confidence_scores": confidence_dict,
        }

    return payload


# ── Selector / trap generators ───────────────────────────


def _filter_valid_selectors(candidate_selectors: list[str]) -> list[str]:
    return [
        sel
        for sel in candidate_selectors
        if "/products/" in sel or "/product/" in sel or "ProductCard" in sel or "product-card" in sel.lower()
    ]


def _generate_selector_changes(
    selector_discovery: dict[str, Any],
    current_site_config: dict[str, Any],
) -> dict[str, Any]:
    changes: dict[str, Any] = {}
    candidates = selector_discovery.get("candidate_selectors", [])
    recommended_layer = selector_discovery.get("recommended_layer", "primary")
    if not candidates:
        return changes

    current_selectors = current_site_config.get("selectors", {}) or {}
    plp = current_selectors.get("plp", {}) or {}
    current_pdp = plp.get("pdp_link_selectors", []) or []
    current_tile = plp.get("tile_selectors", []) or []

    valid = _filter_valid_selectors(candidates)
    if not valid:
        return changes

    if recommended_layer == "primary" and current_pdp != valid:
        changes["selectors.plp"] = {"pdp_link_selectors": {"before": current_pdp, "after": valid[:10]}}
    elif recommended_layer in ("secondary", "tertiary") and current_tile != valid:
        changes["selectors.plp"] = {"tile_selectors": {"before": current_tile, "after": valid[:10]}}

    return changes


def _generate_trap_patterns(
    rejection_stats: dict[str, int],
    current_site_config: dict[str, Any],
) -> dict[str, Any]:
    if rejection_stats.get("double_locale_path", 0) <= 0 and rejection_stats.get("trap_pattern", 0) <= 0:
        return {}

    navigation = current_site_config.get("navigation", {}) or {}
    current_patterns: list[str] = navigation.get("trap_url_patterns", []) or []
    new_patterns: list[str] = []

    if rejection_stats.get("double_locale_path", 0) > 0:
        new_patterns.extend([".*/en-[a-z]{2}/en-int/.*", ".*/en-[a-z]{2}/en-[a-z]{2}/.*"])

    unique = [p for p in new_patterns if p not in current_patterns]
    if unique:
        return {"navigation.trap_url_patterns": {"append": unique}}
    return {}


# ── Risk assessment ───────────────────────────────────────

_RISK_MAP: dict[str, tuple[RiskLevel, str]] = {
    "locale redirect loop": (
        RiskLevel.HIGH,
        "ロケールリダイレクトループが検出されました。LocaleGuard の動作を確認してください。",
    ),
    "primary selector mismatch": (
        RiskLevel.MEDIUM,
        "セレクタの不一致が検出されました。DOM 構造の変化の可能性があります。",
    ),
    "trap page navigation": (
        RiskLevel.MEDIUM,
        "Trap ページが検出されました。trap_url_patterns の追加を検討してください。",
    ),
}


def _assess_risk(root_cause: str) -> RiskAssessment:
    level, notes = _RISK_MAP.get(root_cause, (RiskLevel.LOW, "軽微な調整が推奨されます。"))
    return {"overall": level.value, "notes": notes}


# ── Patch candidate ──────────────────────────────────────


def build_moncler_patch_candidate(
    analysis_payload: dict[str, Any],
    current_site_config: dict[str, Any],
) -> dict[str, Any]:
    run_id = analysis_payload.get("run_id", "unknown")
    site = analysis_payload.get("site", "MONCLER_OFFICIAL")
    timestamp = analysis_payload.get("timestamp", datetime.utcnow().isoformat() + "Z")

    changes: dict[str, Any] = {}
    changes.update(_generate_selector_changes(analysis_payload.get("selector_discovery", {}), current_site_config))
    changes.update(
        _generate_trap_patterns(
            analysis_payload.get("moncler_outcome", {}).get("rejection_stats", {}),
            current_site_config,
        )
    )

    root_cause = analysis_payload.get("self_healing", {}).get("root_cause", "")
    risk = _assess_risk(root_cause)

    return {
        "version": "1.0",
        "target_file": "app/config/sites/overrides.local.json",
        "site": site,
        "run_id": run_id,
        "timestamp": timestamp,
        "summary": "Moncler PLP→PDP 抽出失敗に対する selector/trap パッチ提案",
        "changes": changes,
        "risk_assessment": risk,
    }


# ── Markdown formatting ──────────────────────────────────


def _format_changes_markdown(changes: dict[str, Any]) -> str:
    if not changes:
        return "変更内容はありません。\n\n"
    parts: list[str] = []
    for path, data in changes.items():
        parts.append(f"#### {path}\n\n")
        if isinstance(data, dict):
            if "before" in data and "after" in data:
                parts.append(
                    f"**Before:**\n```json\n{json.dumps(data['before'], ensure_ascii=False, indent=2)}\n```\n\n"
                )
                parts.append(f"**After:**\n```json\n{json.dumps(data['after'], ensure_ascii=False, indent=2)}\n```\n\n")
            elif "append" in data:
                parts.append(
                    f"**Append:**\n```json\n{json.dumps(data['append'], ensure_ascii=False, indent=2)}\n```\n\n"
                )
            elif "remove" in data:
                parts.append(
                    f"**Remove:**\n```json\n{json.dumps(data['remove'], ensure_ascii=False, indent=2)}\n```\n\n"
                )
    return "".join(parts)


def _format_selector_discovery(selector_discovery: dict[str, Any]) -> str:
    candidates = selector_discovery.get("candidate_selectors", [])
    if not candidates:
        return ""
    scores: dict[str, float] = selector_discovery.get("confidence_scores", {})
    lines = [
        "### Selector Discovery 候補\n\n",
        f"**Recommended Layer:** `{selector_discovery.get('recommended_layer', 'N/A')}`\n\n",
        "**Candidate Selectors:**\n",
    ]
    for i, sel in enumerate(candidates[:10], 1):
        lines.append(f"{i}. `{sel}` (confidence: {scores.get(sel, 0.0):.2f})\n")
    lines.append("\n")
    return "".join(lines)


def _format_suggested_actions(self_healing: dict[str, Any]) -> str:
    actions = self_healing.get("suggested_actions", [])
    if not actions:
        return "推奨アクションはありません。\n"
    lines: list[str] = []
    for i, action in enumerate(actions, 1):
        if isinstance(action, dict):
            lines.append(
                f"{i}. **{action.get('description', str(action))}** (Risk: {action.get('risk_level', 'MEDIUM')})\n"
            )
        else:
            lines.append(f"{i}. {action}\n")
    return "".join(lines)


# ── Report generation ────────────────────────────────────


def generate_moncler_patch_markdown(
    run_context: Any,
    analysis_payload: dict[str, Any],
    patch_candidate: dict[str, Any],
) -> Path | None:
    run_id = analysis_payload.get("run_id", "unknown")
    reports_dir = Path("docs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = reports_dir / f"MONCLER_PATCH_{run_id}.md"

    try:
        self_healing = analysis_payload.get("self_healing", {})
        risk = patch_candidate.get("risk_assessment", {})

        content = f"""# Moncler Patch Proposal Report

**Run ID:** `{run_id}`
**Timestamp:** `{analysis_payload.get("timestamp", "N/A")}`
**Current URL:** `{analysis_payload.get("current_url", "N/A")}`

## 失敗の概要

### Self-Healing 分析

**Root Cause:** `{self_healing.get("root_cause", "N/A")}`
**Confidence:** `{self_healing.get("confidence", 0.0):.2f}`

**Analysis:**
{self_healing.get("analysis", "N/A")}

### Moncler Outcome
"""
        outcome = analysis_payload.get("moncler_outcome", {})
        for key in (
            "plp_materialized",
            "tiles_detected",
            "pdp_links_raw",
            "pdp_links_accepted",
            "locale_corrections",
            "trap_detected",
        ):
            content += f"- **{key}:** `{outcome.get(key, 'N/A')}`\n"

        content += f"""
## 提案パッチ

### リスク評価

**Overall Risk:** `{risk.get("overall", "N/A")}`

**Notes:**
{risk.get("notes", "N/A")}

### 変更内容

{_format_changes_markdown(patch_candidate.get("changes", {}))}
{_format_selector_discovery(analysis_payload.get("selector_discovery", {}))}
## 推奨アクション

{_format_suggested_actions(self_healing)}
## 適用方法

1. `patch_candidate.json` を確認
2. `overrides.local.json` の MONCLER_OFFICIAL ブロックを編集
3. `git diff` で変更内容を確認
4. `pytest tests/test_moncler_pdp_url.py` を実行
"""
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info("[PatchBuilder][Moncler] Generated Markdown report: %s", markdown_path)
        return markdown_path
    except OSError as e:
        logger.error("[PatchBuilder][Moncler] Failed to generate Markdown report: %s", e, exc_info=True)
        return None


def save_moncler_patch_files(
    run_context: Any,
    analysis_payload: dict[str, Any],
    patch_candidate: dict[str, Any],
    *,
    generate_markdown: bool = False,
) -> dict[str, Path]:
    run_id = analysis_payload.get("run_id", "unknown")
    sh_dir = Path(run_context.run_path) / "self_healing" / "moncler"
    sh_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    for suffix, data in [("analysis", analysis_payload), ("patch_candidate", patch_candidate)]:
        path = sh_dir / f"{run_id}_{suffix}.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            saved[suffix] = path
            logger.info("[PatchBuilder][Moncler] Saved %s: %s", suffix, path)
        except OSError as e:
            logger.error("[PatchBuilder][Moncler] Failed to save %s: %s", suffix, e, exc_info=True)

    if generate_markdown:
        md_path = generate_moncler_patch_markdown(
            run_context=run_context, analysis_payload=analysis_payload, patch_candidate=patch_candidate
        )
        if md_path:
            saved["markdown"] = md_path

    return saved


async def process_moncler_self_healing_results(
    run_context: Any,
    run_id: str,
    site: str,
    current_url: str,
    moncler_outcome: dict[str, Any],
    self_healing_result: dict[str, Any] | None = None,
    selector_discovery_result: dict[str, Any] | None = None,
    current_site_config: dict[str, Any] | None = None,
    *,
    generate_markdown: bool = False,
) -> dict[str, Path]:
    try:
        if current_site_config is None:
            try:
                overrides_path = Path("app/config/sites/overrides.local.json")
                if overrides_path.exists():
                    with open(overrides_path, encoding="utf-8") as f:
                        current_site_config = json.load(f).get("MONCLER_OFFICIAL", {})
                else:
                    logger.warning("[PatchBuilder][Moncler] overrides.local.json not found, skipping")
                    return {}
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("[PatchBuilder][Moncler] Failed to load overrides.local.json: %s", e, exc_info=True)
                return {}

        analysis_payload = build_moncler_analysis_payload(
            run_id=run_id,
            site=site,
            current_url=current_url,
            moncler_outcome=moncler_outcome,
            self_healing_result=self_healing_result,
            selector_discovery_result=selector_discovery_result,
        )
        patch_candidate = build_moncler_patch_candidate(
            analysis_payload=analysis_payload, current_site_config=current_site_config
        )
        saved_paths = save_moncler_patch_files(
            run_context=run_context,
            analysis_payload=analysis_payload,
            patch_candidate=patch_candidate,
            generate_markdown=generate_markdown,
        )
        logger.info(
            "[PatchBuilder][Moncler] Processed: analysis=%s patch=%s",
            saved_paths.get("analysis"),
            saved_paths.get("patch_candidate"),
        )
        return saved_paths
    except Exception as e:
        logger.error("[PatchBuilder][Moncler] Failed to process self-healing results: %s", e, exc_info=True)
        return {}
