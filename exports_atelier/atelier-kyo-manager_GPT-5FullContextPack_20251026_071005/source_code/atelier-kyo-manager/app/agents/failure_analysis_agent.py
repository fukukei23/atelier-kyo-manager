# ==============================================================================
# ファイル名 (File Name): failure_analysis_agent.py
# レジストリ (Registry): app/agents/failure_analysis_agent.py
# 更新日時 (JST): 2025-10-15 10:00
# バージョン (Version): 8.0.0J (LLM-Production, No Stub)
#
# 変更点
# - スタブ実装を完全撤去。app.utils.ai_llm_controller.AILlmController を使用。
# - app.models.result_models.GenerateResult を共通モデルとして利用。
# - runtime_kwargs.disable_ai_forensics=True で解析をスキップ可能。
# - 例外時でも ai_forensic_report.json を最小限出力（エラー内容のみ）。
# ==============================================================================

from __future__ import annotations
import logging
import traceback
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# --- プロジェクトルートを Python パスへ（堅牢化） ---
try:
    APP_ROOT = Path(__file__).resolve().parents[2]
    if str(APP_ROOT) not in sys.path:
        sys.path.insert(0, str(APP_ROOT))
except Exception:
    pass

from app.core.run_context import RunContext
from app.utils.ai_llm_controller import AILlmController
from app.models.result_models import GenerateResult

logger = logging.getLogger(__name__)


class FailureAnalysisAgent:
    """実行失敗時にフォレンジック（AI 根本原因分析）を行うメタエージェント。"""

    def __init__(self, runtime_kwargs: Optional[Dict[str, Any]] = None):
        self.runtime_kwargs = runtime_kwargs or {}
        self.llm_controller = AILlmController()

    def analyze(
        self,
        *,
        error: Exception,
        site: str,
        site_config: Dict[str, Any],
        html_content: Optional[str],
        run_context: RunContext,
    ) -> GenerateResult:
        """
        与えられた失敗文脈を AI に投げ、提案を受け取り保存する。
        disable_ai_forensics=True が指定されている場合は即時スキップ。
        """
        if self.runtime_kwargs.get("disable_ai_forensics"):
            logger.info("AI Forensics is disabled by runtime flag. Skipping analysis.")
            minimal = {
                "ai_analysis": {
                    "diagnosis": "AI 解析は無効化されています。",
                    "root_cause": "N/A",
                    "recommended_action": "runtime_kwargs.disable_ai_forensics を False にしてください。",
                },
                "evidence_links": {
                    "system_log": str(run_context.get_path("system.log").resolve()),
                    "console_log": str(run_context.get_path("console.log").resolve()),
                    "trace_file": str(run_context.get_path("trace.zip").resolve()),
                    "har_file": str(run_context.get_path("network.har").resolve()),
                    "screenshots_directory": str(run_context.screenshots_path.resolve()),
                    "final_settings": str(run_context.get_path("resolved_settings.json").resolve()),
                },
            }
            run_context.save_json("ai_forensic_report.json", minimal)
            return GenerateResult(text=json.dumps(minimal["ai_analysis"], ensure_ascii=False))

        prompt = self._build_prompt(
            error=error,
            site=site,
            html_snippet=html_content[:8000] if html_content else "N/A",
            run_context=run_context,
            site_config=site_config,
        )

        logger.info(f"AIによる失敗分析を要請します (RunID: {run_context.run_id})")
        try:
            # 本番 LLM 呼び出し（タスク種別は 'analysis'）
            analysis_result = self.llm_controller.generate(prompt, task_type="analysis")

            # 返答は JSON 文字列想定。安全にパースして保存。
            try:
                parsed_analysis = json.loads(analysis_result.text)
            except Exception:
                parsed_analysis = {
                    "diagnosis": "LLM出力がJSONではありませんでした。",
                    "root_cause": "プロンプトまたはモデルの設定問題により非JSON応答。",
                    "recommended_action": "プロンプトのフォーマット指定を強化、あるいはモデルを切り替え。",
                    "raw_text": analysis_result.text,
                }

            forensic_report = {
                "ai_analysis": parsed_analysis,
                "evidence_links": {
                    "system_log": str(run_context.get_path("system.log").resolve()),
                    "console_log": str(run_context.get_path("console.log").resolve()),
                    "trace_file": str(run_context.get_path("trace.zip").resolve()),
                    "har_file": str(run_context.get_path("network.har").resolve()),
                    "screenshots_directory": str(run_context.screenshots_path.resolve()),
                    "final_settings": str(run_context.get_path("resolved_settings.json").resolve()),
                },
            }

            run_context.save_json("ai_forensic_report.json", forensic_report)
            logger.info(f"AI科学捜査レポートを保存しました: {run_context.get_path('ai_forensic_report.json')}")
            return analysis_result

        except Exception as e:
            logger.error(f"AI分析処理に失敗しました (RunID: {run_context.run_id}): {e}", exc_info=True)
            fallback = {
                "ai_analysis": {
                    "diagnosis": "AI 分析処理で例外が発生しました。",
                    "root_cause": str(e),
                    "recommended_action": "API キーとモデル設定を確認し、ネットワーク到達性・レート制限も点検してください。",
                }
            }
            run_context.save_json("ai_forensic_report.json", fallback)
            return GenerateResult(text=json.dumps(fallback["ai_analysis"], ensure_ascii=False))

    def _build_prompt(
        self,
        *,
        error: Exception,
        site: str,
        html_snippet: str,
        run_context: RunContext,
        site_config: Dict[str, Any],
    ) -> str:
        """根本原因分析のための詳細プロンプトを生成（JSON 指定つき）。"""
        error_type = type(error).__name__
        error_message = str(error)
        traceback_info = "".join(traceback.format_tb(error.__traceback__))
        discovery_settings = site_config.get("discovery_settings", {})

        return f"""
# 命令
あなたは Web 自動化システムに精通したデバッグ・アナリストです。
サイト「{site}」の偵察任務でエラーが発生しました。以下の証跡から、根本原因を推定し、解決策を提案してください。
**出力は必ず以下の JSON だけ**を返してください。日本語で簡潔に。

```json
{{
  "diagnosis": "何が起きたか（端的に）",
  "root_cause": "なぜそうなったか（技術的な推定）",
  "recommended_action": "次にとるべき具体策（手順・設定・コード修正の方向性）"
}}"""
