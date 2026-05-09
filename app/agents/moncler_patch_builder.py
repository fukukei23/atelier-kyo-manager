"""
CR-ATELIER-002 Step 7: Moncler site_config パッチ生成ユーティリティ

Self-Healing / Selector Discovery の結果から、site_config へのパッチ候補を生成する。

責務:
- Self-Healing / Selector Discovery の結果を analysis.json 形式に整形
- 現行 overrides.local.json との差分を計算し、patch_candidate.json を生成
- 必要に応じて Markdown レポートを生成
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_moncler_analysis_payload(
    run_id: str,
    site: str,
    current_url: str,
    moncler_outcome: dict[str, Any],
    self_healing_result: dict[str, Any] | None = None,
    selector_discovery_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    CR-ATELIER-002 Step 7-2: Self-Healing 分析結果を analysis.json 形式に整形

    Args:
        run_id: 実行ID
        site: サイトコード（"MONCLER_OFFICIAL"）
        current_url: 現在のURL
        moncler_outcome: Telemetry に保存した outcome 情報
        self_healing_result: Self-Healing Agent の分析結果（オプション）
        selector_discovery_result: Selector Discovery Agent の提案結果（オプション）

    Returns:
        Dict[str, Any]: analysis.json 相当の dict
    """
    payload = {
        "version": "1.0",
        "run_id": run_id,
        "site": site,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "current_url": current_url,
        "moncler_outcome": moncler_outcome,
    }

    # Self-Healing 結果を整形
    if self_healing_result:
        suggested_actions = []
        for action in self_healing_result.get("suggested_actions", []):
            if isinstance(action, str):
                # 文字列の場合は、id と description に分割
                suggested_actions.append(
                    {
                        "id": f"action_{len(suggested_actions)}",
                        "description": action,
                        "risk_level": "MEDIUM",
                    }
                )
            elif isinstance(action, dict):
                suggested_actions.append(action)

        payload["self_healing"] = {
            "analysis": self_healing_result.get("analysis", ""),
            "root_cause": self_healing_result.get("root_cause", ""),
            "suggested_actions": suggested_actions,
            "confidence": self_healing_result.get("confidence", 0.0),
        }

    # Selector Discovery 結果を整形
    if selector_discovery_result:
        candidate_selectors = selector_discovery_result.get("candidate_selectors", [])
        confidence_scores = selector_discovery_result.get("confidence_scores", [])

        # confidence_scores を dict 形式に変換（selector -> score のマッピング）
        confidence_dict = {}
        if isinstance(confidence_scores, list):
            for i, selector in enumerate(candidate_selectors):
                if i < len(confidence_scores):
                    confidence_dict[selector] = confidence_scores[i]
        elif isinstance(confidence_scores, dict):
            confidence_dict = confidence_scores

        payload["selector_discovery"] = {
            "candidate_selectors": candidate_selectors,
            "recommended_layer": selector_discovery_result.get("recommended_layer", "primary"),
            "confidence_scores": confidence_dict,
        }

    return payload


def build_moncler_patch_candidate(
    analysis_payload: dict[str, Any],
    current_site_config: dict[str, Any],
) -> dict[str, Any]:
    """
    CR-ATELIER-002 Step 7-2: パッチ候補ファイルを生成

    Args:
        analysis_payload: build_moncler_analysis_payload() の出力
        current_site_config: 現行 overrides.local.json の MONCLER_OFFICIAL ブロック

    Returns:
        Dict[str, Any]: patch_candidate.json 相当の dict
    """
    run_id = analysis_payload.get("run_id", "unknown")
    site = analysis_payload.get("site", "MONCLER_OFFICIAL")
    timestamp = analysis_payload.get("timestamp", datetime.utcnow().isoformat() + "Z")

    patch = {
        "version": "1.0",
        "target_file": "app/config/sites/overrides.local.json",
        "site": site,
        "run_id": run_id,
        "timestamp": timestamp,
        "summary": "Moncler PLP→PDP 抽出失敗に対する selector/trap パッチ提案",
        "changes": {},
        "risk_assessment": {
            "overall": "MEDIUM",
            "notes": "パッチ適用前に pytest と実ブラウザ検証を推奨。",
        },
    }

    # Selector Discovery の結果からセレクタ変更を抽出
    selector_discovery = analysis_payload.get("selector_discovery", {})
    candidate_selectors = selector_discovery.get("candidate_selectors", [])
    recommended_layer = selector_discovery.get("recommended_layer", "primary")

    if candidate_selectors:
        # 現行のセレクタを取得
        current_selectors = current_site_config.get("selectors", {}) or {}
        plp_selectors = current_selectors.get("plp", {}) or {}
        current_pdp_link_selectors = plp_selectors.get("pdp_link_selectors", []) or []
        current_tile_selectors = plp_selectors.get("tile_selectors", []) or []

        # フィルタリング: /products/ を含むセレクタのみを採用
        valid_candidates = [
            sel
            for sel in candidate_selectors
            if "/products/" in sel or "/product/" in sel or "ProductCard" in sel or "product-card" in sel.lower()
        ]

        if valid_candidates:
            # recommended_layer に応じて、適切なフィールドに設定
            if recommended_layer == "primary":
                # Primary layer の場合は pdp_link_selectors を更新
                if current_pdp_link_selectors != valid_candidates:
                    patch["changes"]["selectors.plp"] = patch["changes"].get("selectors.plp", {})
                    patch["changes"]["selectors.plp"]["pdp_link_selectors"] = {
                        "before": current_pdp_link_selectors,
                        "after": valid_candidates[:10],  # 最大10件
                    }
            elif recommended_layer in ("secondary", "tertiary"):
                # Secondary/Tertiary の場合は、tile_selectors も更新を検討
                if current_tile_selectors != valid_candidates:
                    patch["changes"]["selectors.plp"] = patch["changes"].get("selectors.plp", {})
                    patch["changes"]["selectors.plp"]["tile_selectors"] = {
                        "before": current_tile_selectors,
                        "after": valid_candidates[:10],  # 最大10件
                    }

    # Self-Healing の結果から trap_url_patterns の追加を抽出
    self_healing = analysis_payload.get("self_healing", {})
    root_cause = self_healing.get("root_cause", "")
    moncler_outcome = analysis_payload.get("moncler_outcome", {})
    rejection_stats = moncler_outcome.get("rejection_stats", {})

    # double_locale_path や trap_pattern が多かった場合、trap_url_patterns を追加
    if rejection_stats.get("double_locale_path", 0) > 0 or rejection_stats.get("trap_pattern", 0) > 0:
        current_navigation = current_site_config.get("navigation", {}) or {}
        current_trap_patterns = current_navigation.get("trap_url_patterns", []) or []

        # 追加すべき trap パターンを提案
        new_trap_patterns = []
        if rejection_stats.get("double_locale_path", 0) > 0:
            # 二重ロケールパターンを追加
            new_trap_patterns.extend(
                [
                    ".*/en-[a-z]{2}/en-int/.*",
                    ".*/en-[a-z]{2}/en-[a-z]{2}/.*",
                ]
            )

        if new_trap_patterns:
            # 既存のパターンと重複を排除
            unique_new_patterns = [p for p in new_trap_patterns if p not in current_trap_patterns]

            if unique_new_patterns:
                patch["changes"]["navigation.trap_url_patterns"] = {
                    "append": unique_new_patterns,
                }

    # リスク評価を更新
    if root_cause == "locale redirect loop":
        patch["risk_assessment"]["overall"] = "HIGH"
        patch["risk_assessment"]["notes"] = (
            "ロケールリダイレクトループが検出されました。LocaleGuard の動作を確認してください。"
        )
    elif root_cause == "primary selector mismatch":
        patch["risk_assessment"]["overall"] = "MEDIUM"
        patch["risk_assessment"]["notes"] = "セレクタの不一致が検出されました。DOM 構造の変化の可能性があります。"
    elif root_cause == "trap page navigation":
        patch["risk_assessment"]["overall"] = "MEDIUM"
        patch["risk_assessment"]["notes"] = "Trap ページが検出されました。trap_url_patterns の追加を検討してください。"
    else:
        patch["risk_assessment"]["overall"] = "LOW"
        patch["risk_assessment"]["notes"] = "軽微な調整が推奨されます。"

    return patch


def save_moncler_patch_files(
    run_context: Any,
    analysis_payload: dict[str, Any],
    patch_candidate: dict[str, Any],
    *,
    generate_markdown: bool = False,
) -> dict[str, Path]:
    """
    CR-ATELIER-002 Step 7-3: パッチファイルを保存

    Args:
        run_context: RunContext オブジェクト
        analysis_payload: build_moncler_analysis_payload() の出力
        patch_candidate: build_moncler_patch_candidate() の出力
        generate_markdown: Markdown レポートを生成するか（オプション）

    Returns:
        Dict[str, Path]: 保存されたファイルのパス
            - "analysis": analysis.json のパス
            - "patch_candidate": patch_candidate.json のパス
            - "markdown": markdown レポートのパス（generate_markdown=True の場合）
    """
    run_id = analysis_payload.get("run_id", "unknown")

    # instance/self_healing/moncler/ ディレクトリを作成
    self_healing_dir = Path(run_context.run_path) / "self_healing" / "moncler"
    self_healing_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = {}

    # analysis.json を保存
    analysis_path = self_healing_dir / f"{run_id}_analysis.json"
    try:
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(analysis_payload, f, ensure_ascii=False, indent=2)
        saved_paths["analysis"] = analysis_path
        logger.info(f"[PatchBuilder][Moncler] Saved analysis.json: {analysis_path}")
    except Exception as e:
        logger.error(f"[PatchBuilder][Moncler] Failed to save analysis.json: {e}", exc_info=True)

    # patch_candidate.json を保存
    patch_path = self_healing_dir / f"{run_id}_patch_candidate.json"
    try:
        with open(patch_path, "w", encoding="utf-8") as f:
            json.dump(patch_candidate, f, ensure_ascii=False, indent=2)
        saved_paths["patch_candidate"] = patch_path
        logger.info(f"[PatchBuilder][Moncler] Saved patch_candidate.json: {patch_path}")
    except Exception as e:
        logger.error(f"[PatchBuilder][Moncler] Failed to save patch_candidate.json: {e}", exc_info=True)

    # Markdown レポートを生成（オプション）
    if generate_markdown:
        markdown_path = generate_moncler_patch_markdown(
            run_context=run_context,
            analysis_payload=analysis_payload,
            patch_candidate=patch_candidate,
        )
        if markdown_path:
            saved_paths["markdown"] = markdown_path

    return saved_paths


def generate_moncler_patch_markdown(
    run_context: Any,
    analysis_payload: dict[str, Any],
    patch_candidate: dict[str, Any],
) -> Path | None:
    """
    CR-ATELIER-002 Step 7-3: Markdown レポートを生成

    Args:
        run_context: RunContext オブジェクト
        analysis_payload: build_moncler_analysis_payload() の出力
        patch_candidate: build_moncler_patch_candidate() の出力

    Returns:
        Optional[Path]: 生成された Markdown ファイルのパス（失敗時は None）
    """
    run_id = analysis_payload.get("run_id", "unknown")

    # docs/reports/ ディレクトリを作成
    reports_dir = Path("docs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = reports_dir / f"MONCLER_PATCH_{run_id}.md"

    try:
        # Markdown コンテンツを生成
        moncler_outcome = analysis_payload.get("moncler_outcome", {})
        self_healing = analysis_payload.get("self_healing", {})
        selector_discovery = analysis_payload.get("selector_discovery", {})
        changes = patch_candidate.get("changes", {})
        risk_assessment = patch_candidate.get("risk_assessment", {})

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

- **PLP Materialized:** `{moncler_outcome.get("plp_materialized", False)}`
- **Tiles Detected:** `{moncler_outcome.get("tiles_detected", 0)}`
- **PDP Links Raw:** `{moncler_outcome.get("pdp_links_raw", 0)}`
- **PDP Links Accepted:** `{moncler_outcome.get("pdp_links_accepted", 0)}`
- **Locale Corrections:** `{moncler_outcome.get("locale_corrections", 0)}`
- **Trap Detected:** `{moncler_outcome.get("trap_detected", False)}`

## 提案パッチ

### リスク評価

**Overall Risk:** `{risk_assessment.get("overall", "N/A")}`

**Notes:**
{risk_assessment.get("notes", "N/A")}

### 変更内容

"""

        # changes を Markdown 形式で出力
        if changes:
            for change_path, change_data in changes.items():
                content += f"#### {change_path}\n\n"

                if isinstance(change_data, dict):
                    if "before" in change_data and "after" in change_data:
                        # before/after 形式
                        content += "**Before:**\n"
                        content += "```json\n"
                        content += json.dumps(change_data["before"], ensure_ascii=False, indent=2)
                        content += "\n```\n\n"
                        content += "**After:**\n"
                        content += "```json\n"
                        content += json.dumps(change_data["after"], ensure_ascii=False, indent=2)
                        content += "\n```\n\n"
                    elif "append" in change_data:
                        # append 形式
                        content += "**Append:**\n"
                        content += "```json\n"
                        content += json.dumps(change_data["append"], ensure_ascii=False, indent=2)
                        content += "\n```\n\n"
                    elif "remove" in change_data:
                        # remove 形式
                        content += "**Remove:**\n"
                        content += "```json\n"
                        content += json.dumps(change_data["remove"], ensure_ascii=False, indent=2)
                        content += "\n```\n\n"
        else:
            content += "変更内容はありません。\n\n"

        # Selector Discovery の候補セレクタを表示
        if selector_discovery.get("candidate_selectors"):
            content += "### Selector Discovery 候補\n\n"
            content += "**Recommended Layer:** `{}`\n\n".format(selector_discovery.get("recommended_layer", "N/A"))
            content += "**Candidate Selectors:**\n"
            for i, selector in enumerate(selector_discovery.get("candidate_selectors", [])[:10], 1):
                confidence = selector_discovery.get("confidence_scores", {}).get(selector, 0.0)
                content += f"{i}. `{selector}` (confidence: {confidence:.2f})\n"
            content += "\n"

        # 推奨アクション
        content += "## 推奨アクション\n\n"
        suggested_actions = self_healing.get("suggested_actions", [])
        if suggested_actions:
            for i, action in enumerate(suggested_actions, 1):
                if isinstance(action, dict):
                    action_desc = action.get("description", str(action))
                    risk_level = action.get("risk_level", "MEDIUM")
                    content += f"{i}. **{action_desc}** (Risk: {risk_level})\n"
                else:
                    content += f"{i}. {action}\n"
        else:
            content += "推奨アクションはありません。\n"

        content += "\n## 適用方法\n\n"
        content += "1. `patch_candidate.json` を確認\n"
        content += "2. `overrides.local.json` の MONCLER_OFFICIAL ブロックを編集\n"
        content += "3. `git diff` で変更内容を確認\n"
        content += "4. `pytest tests/test_moncler_pdp_url.py` を実行\n"
        content += '5. 実ブラウザ検証: `python -m app.scripts.run_site moncler --query "down jacket" --headful`\n'

        # ファイルに保存
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"[PatchBuilder][Moncler] Generated Markdown report: {markdown_path}")
        return markdown_path

    except Exception as e:
        logger.error(f"[PatchBuilder][Moncler] Failed to generate Markdown report: {e}", exc_info=True)
        return None


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
    """
    CR-ATELIER-002 Step 7-4: Self-Healing 結果を処理してパッチファイルを生成・保存

    この関数は NavigationDriver から呼び出されることを想定しています。

    Args:
        run_context: RunContext オブジェクト
        run_id: 実行ID
        site: サイトコード（"MONCLER_OFFICIAL"）
        current_url: 現在のURL
        moncler_outcome: Telemetry に保存した outcome 情報
        self_healing_result: Self-Healing Agent の分析結果（オプション）
        selector_discovery_result: Selector Discovery Agent の提案結果（オプション）
        current_site_config: 現行 overrides.local.json の MONCLER_OFFICIAL ブロック（オプション）
        generate_markdown: Markdown レポートを生成するか（オプション）

    Returns:
        Dict[str, Path]: 保存されたファイルのパス
    """
    try:
        # current_site_config が提供されていない場合、overrides.local.json から読み込む
        if current_site_config is None:
            try:
                from pathlib import Path

                overrides_path = Path("app/config/sites/overrides.local.json")
                if overrides_path.exists():
                    with open(overrides_path, encoding="utf-8") as f:
                        overrides_data = json.load(f)
                        current_site_config = overrides_data.get("MONCLER_OFFICIAL", {})
                else:
                    logger.warning(
                        "[PatchBuilder][Moncler] overrides.local.json not found, skipping patch candidate generation"
                    )
                    return {}
            except Exception as e:
                logger.warning(f"[PatchBuilder][Moncler] Failed to load overrides.local.json: {e}", exc_info=True)
                return {}

        # analysis.json を生成
        analysis_payload = build_moncler_analysis_payload(
            run_id=run_id,
            site=site,
            current_url=current_url,
            moncler_outcome=moncler_outcome,
            self_healing_result=self_healing_result,
            selector_discovery_result=selector_discovery_result,
        )

        # patch_candidate.json を生成
        patch_candidate = build_moncler_patch_candidate(
            analysis_payload=analysis_payload,
            current_site_config=current_site_config,
        )

        # ファイルを保存
        saved_paths = save_moncler_patch_files(
            run_context=run_context,
            analysis_payload=analysis_payload,
            patch_candidate=patch_candidate,
            generate_markdown=generate_markdown,
        )

        logger.info(
            f"[PatchBuilder][Moncler] Processed self-healing results: "
            f"analysis={saved_paths.get('analysis')}, "
            f"patch={saved_paths.get('patch_candidate')}"
        )

        return saved_paths

    except Exception as e:
        logger.error(f"[PatchBuilder][Moncler] Failed to process self-healing results: {e}", exc_info=True)
        return {}
