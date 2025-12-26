#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CR-ATELIER-003 Phase D-7: Self-Healing パッチ適用スクリプト（オフライン実行用）

使用例:
    python scripts/apply_self_healing_patch.py \
        --run-id "RUN_20251210_XXXXXX" \
        --site "MONCLER_OFFICIAL" \
        --overrides "app/config/sites/overrides.local.json"
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.self_healing_patch_applier import SelfHealingPatchApplier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Apply self-healing patch candidate to overrides.local.json"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run ID (e.g., RUN_20251210_XXXXXX)",
    )
    parser.add_argument(
        "--site",
        type=str,
        required=True,
        help="Site code (e.g., MONCLER_OFFICIAL)",
    )
    parser.add_argument(
        "--overrides",
        type=str,
        default="app/config/sites/overrides.local.json",
        help="Path to overrides.local.json (default: app/config/sites/overrides.local.json)",
    )
    parser.add_argument(
        "--candidate-path",
        type=str,
        default=None,
        help="Path to patch_candidate_self_healing.json (default: instance/runs/<run_id>/patch_candidate_self_healing.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode (do not actually apply the patch)",
    )
    
    args = parser.parse_args()
    
    # パスを解決
    overrides_path = Path(args.overrides)
    if not overrides_path.is_absolute():
        overrides_path = PROJECT_ROOT / overrides_path
    
    if args.candidate_path:
        candidate_path = Path(args.candidate_path)
        if not candidate_path.is_absolute():
            candidate_path = PROJECT_ROOT / candidate_path
    else:
        # デフォルトパス: instance/runs/<run_id>/patch_candidate_self_healing.json
        candidate_path = PROJECT_ROOT / "instance" / "runs" / args.run_id / "patch_candidate_self_healing.json"
    
    # ファイルの存在確認
    if not candidate_path.exists():
        logger.error(f"Patch candidate file not found: {candidate_path}")
        logger.info(f"Expected path: instance/runs/{args.run_id}/patch_candidate_self_healing.json")
        return 1
    
    if not overrides_path.exists():
        logger.error(f"Overrides file not found: {overrides_path}")
        return 1
    
    # パッチ候補を読み込んで確認
    try:
        with open(candidate_path, "r", encoding="utf-8") as f:
            patch_candidate = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in patch candidate: {e}")
        return 1
    
    target_site = patch_candidate.get("target_site", "unknown")
    if target_site != args.site:
        logger.warning(
            f"Patch candidate target_site ({target_site}) does not match --site ({args.site})"
        )
    
    changes = patch_candidate.get("changes", [])
    if not changes:
        logger.warning("Patch candidate has no changes to apply")
        return 0
    
    # 適用前の確認メッセージ
    logger.info("=" * 60)
    logger.info("Self-Healing Patch Application")
    logger.info("=" * 60)
    logger.info(f"Run ID: {args.run_id}")
    logger.info(f"Target Site: {target_site}")
    logger.info(f"Overrides File: {overrides_path}")
    logger.info(f"Patch Candidate: {candidate_path}")
    logger.info(f"Changes Count: {len(changes)}")
    logger.info("")
    logger.info("Changes to apply:")
    for i, change in enumerate(changes, 1):
        logger.info(f"  {i}. {change.get('path')} - {change.get('action')}")
    logger.info("")
    
    if args.dry_run:
        logger.info("[DRY RUN] Patch would be applied, but --dry-run is set. Exiting.")
        return 0
    
    # パッチを適用
    applier = SelfHealingPatchApplier()
    result = applier.apply_patch_candidate(
        candidate_path=candidate_path,
        overrides_path=overrides_path,
    )
    
    if result["applied"]:
        logger.info("=" * 60)
        logger.info("Patch Applied Successfully")
        logger.info("=" * 60)
        logger.info(f"Backup File: {result['backup_path']}")
        logger.info(f"Applied Diff Operations: {len(result['diff_ops'])}")
        logger.info("")
        logger.info("Next Steps:")
        logger.info("  1. Review the changes in overrides.local.json")
        logger.info("  2. Run pytest to verify the changes")
        logger.info("  3. Test with a real browser run")
        logger.info("  4. If issues occur, restore from backup:")
        logger.info(f"     cp {result['backup_path']} {overrides_path}")
        return 0
    else:
        logger.error("=" * 60)
        logger.error("Patch Application Failed")
        logger.error("=" * 60)
        logger.error(f"Error: {result['error']}")
        if result.get("backup_path") and result["backup_path"].exists():
            logger.info(f"Backup file was created: {result['backup_path']}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

