# ==============================================================================
# ファイル名 (File Name): reporting_agent.py
# レジストリ (Registry): app/agents/reporting_agent.py
# 更新日時 (Date & Time JST): 2025年09月21日 20:02:00
# バージョン (Version): 3.0.0J (Intelligence Briefing)
#
# --- 操作するソフト/前提 (Software & Prerequisites) ---
# - Python 3.10以上
# - 依存モジュール: app/models/result_models.py
#
# --- 使用方法 (Usage) ---
# このエージェントは、Orchestratorから作戦完了後に呼び出されます。
# 蓄積されたデータから日次レポートを生成する能力と、単一の作戦結果から
# 即時サマリーを生成する能力を持ちます。
#
# --- v3.0.0Jでの主な変更点 ---
# - [インテリジェンス統合] `build_run_summary_report`が、失敗した任務の
#   `ai_forensic_report.json`を自動的に読み込み、AIによる失敗分析
#   （診断、原因、対策）をレポートに直接含めるようになりました。
# - [視認性向上] HTMLレポートのデザインを全面的に刷新し、よりモダンで
#   可読性の高いレイアウトに変更しました。
# ==============================================================================
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# --- プロジェクトルートをPythonの検索パスに追加 ---
APP_ROOT = Path(__file__).resolve().parents[2]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.models.result_models import DiscoveryResult

logger = logging.getLogger(__name__)


class ReportingAgent:
    """
    作戦結果を集約・分析し、人間が意思決定を行うためのレポートを生成する情報将校。
    """

    def __init__(self, use_db: bool = False, input_dir: Path = Path("output")):
        self.use_db = use_db  # NOTE: DB関連のロジックは今回変更なし
        self.input_dir = input_dir
        self.output_dir = Path("output/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_run_summary_report(self, results: list[Any]) -> Path:
        """
        単一の作戦実行結果(results)から、概要をまとめたHTMLレポートを生成する。
        """
        run_id = "N/A"
        for r in results:
            if isinstance(r, DiscoveryResult) and r.evidence and r.evidence.get("run_id"):
                run_id = r.evidence.get("run_id")
                break

        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        successful_runs = [r for r in results if isinstance(r, DiscoveryResult) and r.ok]
        failed_runs = [r for r in results if isinstance(r, DiscoveryResult) and not r.ok]

        # --- HTMLコンテンツの生成 ---
        success_html = self._render_success_details(successful_runs)
        failures_html = self._render_failure_details(failed_runs)

        html = f"""
<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>作戦サマリーレポート - {run_id}</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, 'Noto Sans', sans-serif; margin: 0; background-color: #f9fafb; color: #1f2937; }}
    .container {{ max-width: 900px; margin: 2rem auto; padding: 2rem; background-color: #ffffff; border-radius: 0.5rem; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06); }}
    h1, h2 {{ color: #111827; border-bottom: 1px solid #e5e7eb; padding-bottom: 0.5rem; }}
    h1 {{ font-size: 1.875rem; }} h2 {{ font-size: 1.5rem; margin-top: 2rem; }}
    .meta-info {{ background-color: #f3f4f6; padding: 1rem; border-radius: 0.375rem; margin-bottom: 2rem; font-size: 0.875rem; }}
    .run-item {{ border: 1px solid #e5e7eb; border-radius: 0.375rem; margin-bottom: 1rem; overflow: hidden; }}
    .run-header {{ padding: 0.75rem 1rem; font-weight: bold; }}
    .run-header.success {{ background-color: #d1fae5; color: #065f46; }}
    .run-header.failure {{ background-color: #fee2e2; color: #991b1b; }}
    .run-body {{ padding: 1rem; font-size: 0.875rem; line-height: 1.5; }}
    .run-body strong {{ color: #374151; }}
    .ai-analysis {{ background-color: #fefce8; border: 1px solid #fde68a; border-radius: 0.25rem; margin-top: 1rem; padding: 1rem; }}
    .ai-analysis dl {{ margin: 0; }} .ai-analysis dt {{ font-weight: bold; color: #713f12; margin-top: 0.5rem; }} .ai-analysis dd {{ margin-left: 1.5rem; }}
    a {{ color: #2563eb; text-decoration: none; }} a:hover {{ text-decoration: underline; }}
</style>
</head><body><div class="container">
<h1>作戦サマリーレポート</h1>
<div class="meta-info">
    <b>Run ID (代表):</b> {run_id}<br>
    <b>生成時刻:</b> {start_time}
</div>
<h2>任務結果概要</h2>
{success_html}
{failures_html}
</div></body></html>"""

        report_filename = f"run_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report_path = self.output_dir / report_filename
        report_path.write_text(html, encoding="utf-8")
        return report_path

    def _render_success_details(self, successful_runs: list[DiscoveryResult]) -> str:
        if not successful_runs:
            return "<h2>成功した任務 (0件)</h2>"
        items_html = ""
        for r in successful_runs:
            items_html += f"""
            <div class="run-item">
                <div class="run-header success">✔ SUCCESS: {r.site}</div>
                <div class="run-body"><strong>メッセージ:</strong> {r.message}</div>
            </div>"""
        return f"<h2>成功した任務 ({len(successful_runs)}件)</h2>{items_html}"

    def _render_failure_details(self, failed_runs: list[DiscoveryResult]) -> str:
        if not failed_runs:
            return "<h2>失敗した任務 (0件)</h2>"
        items_html = ""
        for r in failed_runs:
            run_id = (r.evidence or {}).get("run_id", "")
            ai_analysis_html = self._get_ai_analysis_html(run_id)

            items_html += f"""
            <div class="run-item">
                <div class="run-header failure">✖ FAILURE: {r.site}</div>
                <div class="run-body">
                    <strong>メッセージ:</strong> {r.message}<br>
                    <strong>詳細な証跡:</strong> <a href="../../instance/runs/{run_id}" target="_blank">作戦記録フォルダ (RunID: {run_id})</a>
                    {ai_analysis_html}
                </div>
            </div>"""
        return f"<h2>失敗した任務 ({len(failed_runs)}件)</h2>{items_html}"

    def _get_ai_analysis_html(self, run_id: str) -> str:
        """指定されたrun_idのAI科学捜査レポートを読み込み、HTMLを生成する。"""
        if not run_id:
            return ""
        try:
            forensic_report_path = Path(f"instance/runs/{run_id}/ai_forensic_report.json")
            if forensic_report_path.exists():
                report_data = json.loads(forensic_report_path.read_text(encoding="utf-8"))
                ai_analysis = report_data.get("ai_analysis", {})
                if ai_analysis:
                    return f"""
                    <div class="ai-analysis">
                        <dl>
                            <dt>AIによる診断:</dt><dd>{ai_analysis.get("diagnosis", "N/A")}</dd>
                            <dt>根本原因の推測:</dt><dd>{ai_analysis.get("root_cause", "N/A")}</dd>
                            <dt>推奨される対策:</dt><dd>{ai_analysis.get("recommended_action", "N/A")}</dd>
                        </dl>
                    </div>"""
        except Exception as e:
            logger.warning(f"RunID {run_id} のAI分析レポートの読み込みに失敗: {e}")
        return ""

    # ... (build_daily_reportなどの既存メソッドは変更なし) ...
