# -*- coding: utf-8 -*-
# ==============================================================================
# File: app/utils/diagnostics.py
# Version: 1.1.0J (Auto Bundle + Manifest / Final)
# Date (JST): 2025-10-20 00:30
#
# 使い方:
#   from app.utils.diagnostics import make_diagnostic_bundle
#   bundle_zip = await make_diagnostic_bundle(run_context, include_trace_report=True)
# ==============================================================================

from __future__ import annotations
import asyncio
import json
import shutil
from pathlib import Path
from typing import List, Optional

from app.utils.trace_reporter import make_trace_report

BUNDLE_FILES_GLOBS = [
    "ai_forensic_report.json",
    "navigation_plan.json",
    "raw_hrefs_raw_abs.json",
    "selector_counts_*.json",
    "pdp_extracted_data.json",
    "plp_extracted_items.json",
    "fail_snapshot.md",
    "trace.zip",
    "trace_report.md",
    "network.har",
    "screenshots/*.png",
    "trace_shot_*.png",
]

async def make_diagnostic_bundle(run_context, include_trace_report: bool = True, max_trace_images: int = 6) -> str:
    run_dir = Path(run_context.run_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    # trace_report.md と trace_shot_*.png を生成（あれば上書きOK）
    trace_zip = run_dir / "trace.zip"
    if include_trace_report and trace_zip.exists():
        try:
            make_trace_report(str(trace_zip), out_dir=str(run_dir), max_images=max_trace_images)
        except Exception as e:
            run_context.logger.warning(f"[diag] make_trace_report failed: {e}")

    # 収集対象のファイルを列挙
    collected: List[str] = []
    for pat in BUNDLE_FILES_GLOBS:
        for p in run_dir.glob(pat):
            if p.is_file():
                collected.append(str(p.relative_to(run_dir)))

    # MANIFEST.json
    manifest = {
        "run_id": getattr(run_context, "run_id", "unknown"),
        "base": str(run_dir.resolve()),
        "files": collected,
    }
    (run_dir / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # README.md
    (run_dir / "README.md").write_text(
        "# Diagnostic Bundle\n\n"
        f"- Run ID: `{getattr(run_context, 'run_id', 'unknown')}`\n"
        "- 解凍後は MANIFEST.json の `files` を辿れば主要アーティファクトにアクセスできます。\n"
        "- 重要: `trace_report.md` はスクショと簡易統計をまとめたものです。\n",
        encoding="utf-8"
    )

    # Zip化
    out_zip = run_dir.with_suffix(".zip")
    if out_zip.exists():
        out_zip.unlink()
    shutil.make_archive(str(out_zip.with_suffix("")), "zip", run_dir)
    run_context.logger.info(f"[diag] bundle created: {out_zip}")
    return str(out_zip)
