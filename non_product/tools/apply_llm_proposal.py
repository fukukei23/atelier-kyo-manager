#!/usr/bin/env python3
"""
半自動パイプライン用のユーティリティ。

exports/llm_proposals に保存された LLM 提案(JSON)を安全に overrides.local.json へ適用する。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _load_apply_selector_proposal():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "app.utils.overrides_store", ROOT / "app" / "utils" / "overrides_store.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("Failed to load app.utils.overrides_store")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "apply_selector_proposal")


apply_selector_proposal = _load_apply_selector_proposal()


def _find_latest_proposal(site: str, base_dir: Path) -> Optional[Path]:
    site_slug = site.lower()
    candidates = sorted(
        base_dir.glob(f"{site_slug}_*_proposal.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def run(args: argparse.Namespace) -> Tuple[bool, Optional[str]]:
    base_dir = Path(args.base_dir)
    proposal_path = Path(args.proposal) if args.proposal else None

    if not proposal_path:
        proposal_path = _find_latest_proposal(args.site, base_dir)
        if not proposal_path:
            print(f"[apply_llm_proposal] 提案ファイルが見つかりません: {base_dir} ({args.site})")
            return False, None

    overrides_path = Path(args.overrides)
    success, diff = apply_selector_proposal(
        proposal_path=proposal_path,
        overrides_path=overrides_path,
        site=args.site,
    )
    if success:
        print(f"[apply_llm_proposal] 適用完了: {proposal_path} -> {overrides_path}")
        if diff:
            print(diff)
    else:
        print(f"[apply_llm_proposal] 適用に失敗: {proposal_path}")
    return success, diff


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM提案を overrides.local.json にマージする半自動適用ツール。")
    parser.add_argument("--site", required=True, help="サイトキー（例: MONCLER_OFFICIAL）")
    parser.add_argument(
        "--proposal",
        help="適用したい提案JSONへのパス。未指定の場合は exports/llm_proposals から最新を選択。",
    )
    parser.add_argument(
        "--overrides",
        default="app/config/sites/overrides.local.json",
        help="マージ先 overrides ファイルのパス。",
    )
    parser.add_argument(
        "--base-dir",
        default="exports/llm_proposals",
        help="提案JSONを探索するディレクトリ。",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
