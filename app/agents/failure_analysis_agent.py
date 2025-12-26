# 2025年09月21日 15:22 JST
# ファイル名: failure_analysis_agent.py
# レジストリ: app/agents/failure_analysis_agent.py
# バージョン: 7.0.0J (Forensic Reporting)
#
# --- v7.0.0Jでの主な変更点 ---
# - [統合フォレンジックレポート] あなたの指摘に基づき、AIによる分析結果を
#   保存する際、その作戦に関連する全ての主要な証跡（Trace, HAR, Logs,
#   Screenshots）へのパスも併記するようにしました。
# - [マスターインデックス化] これにより`ai_analysis.json`は、インシデントの
#   根本原因と、その証拠一式を網羅した「科学捜査報告書」として機能し、
#   後続の分析や報告を劇的に効率化します。

from __future__ import annotations
import logging
import traceback
import json
from typing import Optional, Dict, Any
import sys
from pathlib import Path

# --- プロジェクトルートをPythonの検索パスに追加 ---
APP_ROOT = Path(__file__).resolve().parents[2]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

# --- プロジェクトのモジュールをインポート ---
try:
    from app.core.run_context import RunContext
except Exception:
    from core.run_context import RunContext
# from app.utils.ai_llm_controller import AILlmController
# from app.models.result_models import GenerateResult

# --- スタブクラス (開発用) ---
class GenerateResult:
    def __init__(self, text: str, cost_usd: float = 0.0):
        self.text = text
        self.cost_usd = cost_usd

class AILlmController:
    def generate(self, prompt: str, task_type: str) -> GenerateResult:
        logging.info("AILlmController (Stub): AIへのリクエストをシミュレートします。")
        dummy_response = {
          "diagnosis": "ナビゲーション中にタイムアウトが発生しました。",
          "root_cause": "ターゲットサイトの応答が遅いか、表示を待機すべき特定の要素（セレクタ）がDOM上に存在しなかった可能性があります。",
          "recommended_action": "config/overrides.local.jsonの該当サイト設定で、`timeout_sec`を延長するか、`wait_for_selectors`の値を見直してください。"
        }
        return GenerateResult(text=json.dumps(dummy_response, ensure_ascii=False, indent=2))
# --- スタブここまで ---

logger = logging.getLogger(__name__)

class FailureAnalysisAgent:
    """エージェントの失敗を分析し、原因と対策を提案するメタエージェント。"""
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
        run_context: RunContext
    ) -> GenerateResult:
        """
        与えられた失敗コンテキストを分析し、AIの分析結果を返す。
        分析結果は全ての証跡へのリンクと共にRunContextに保存される。
        """
        prompt = self._build_prompt(
            error=error,
            site=site,
            html_snippet=html_content[:8000] if html_content else "N/A",
            run_context=run_context,
            site_config=site_config
        )

        logger.info(f"AIによる失敗分析を要請します (RunID: {run_context.run_id})")
        try:
            analysis_result = self.llm_controller.generate(prompt, task_type="analysis")

            # --- ★★★ 統合フォレンジックレポートを作成 ★★★ ---
            parsed_analysis = json.loads(analysis_result.text)

            forensic_report = {
                "ai_analysis": parsed_analysis,
                "evidence_links": {
                    "system_log": str(run_context.get_path("system.log").resolve()),
                    "console_log": str(run_context.get_path("console.log").resolve()),
                    "trace_file": str(run_context.get_path("trace.zip").resolve()),
                    "har_file": str(run_context.get_path("network.har").resolve()),
                    "screenshots_directory": str(run_context.screenshots_path.resolve()),
                    "final_settings": str(run_context.get_path("resolved_settings.json").resolve())
                }
            }

            run_context.save_json("ai_forensic_report.json", forensic_report)
            logger.info(f"AI科学捜査レポートを保存しました: {run_context.get_path('ai_forensic_report.json')}")

            return analysis_result
        except Exception as e:
            logger.error(f"AIによる分析処理自体が失敗しました (RunID: {run_context.run_id}): {e}", exc_info=True)
            error_text = f'{{"error": "AI analysis process failed.", "details": "{str(e)}"}}'
            return GenerateResult(text=error_text)

    async def analyze_failure_context(
        self,
        failure_context: Dict[str, Any],
        *,
        run_context: Optional[RunContext] = None,
    ) -> Dict[str, Any]:
        """
        CR-ATELIER-003 Phase D-5: failure_context を直接受け取って分析する
        
        BrowserOrchestrator から呼び出されることを想定したメソッド。
        failure_context から必要な情報を抽出し、既存の analyze メソッドを呼び出す。
        
        Args:
            failure_context: 標準化された failure_context 辞書
            run_context: RunContext オブジェクト（オプション、failure_context に含まれている場合は不要）
        
        Returns:
            分析結果の辞書。以下のフィールドを含む:
                - summary: 人間向け要約
                - root_causes: 想定される原因リスト
                - suggested_fixes: 修正案のリスト
                - confidence: 信頼度（0.0-1.0）
        """
        try:
            # failure_context から必要な情報を抽出
            site = failure_context.get("site", "unknown")
            error_type = failure_context.get("error_type", "unknown")
            error_class = failure_context.get("error_class", "UnknownError")
            error_message = failure_context.get("error_message", "")
            site_config_summary = failure_context.get("site_config_summary", {})
            site_config = {
                "site_code": site_config_summary.get("site_code", site),
                "selectors": {
                    "plp": {} if not site_config_summary.get("has_plp_selectors") else {},
                    "pdp": {} if not site_config_summary.get("has_pdp_selectors") else {},
                },
            }
            
            # run_context が提供されていない場合、failure_context から取得を試みる
            if run_context is None:
                run_id = failure_context.get("run_id", "unknown")
                # RunContext の最小限の実装を作成（analyze メソッドが要求するため）
                class MinimalRunContext:
                    def __init__(self, run_id: str):
                        self.run_id = run_id
                    def get_path(self, name: str) -> Path:
                        return Path(f"/tmp/{run_id}/{name}")
                    def save_json(self, name: str, data: Any) -> None:
                        pass
                    @property
                    def screenshots_path(self) -> Path:
                        return Path(f"/tmp/{run_id}/screenshots")
                
                run_context = MinimalRunContext(run_id)
            
            # エラーオブジェクトを再構築（既存の analyze メソッドが要求するため）
            class ReconstructedError(Exception):
                def __init__(self, error_class: str, error_message: str):
                    super().__init__(error_message)
                    self.__class__.__name__ = error_class
            
            error = ReconstructedError(error_class, error_message)
            
            # HTML コンテンツを取得（可能であれば）
            html_content = None
            dom_snapshot_path = failure_context.get("dom_snapshot_path")
            if dom_snapshot_path and Path(dom_snapshot_path).exists():
                try:
                    with open(dom_snapshot_path, "r", encoding="utf-8") as f:
                        html_content = f.read()
                except Exception:
                    pass
            
            # 既存の analyze メソッドを呼び出す
            analysis_result = self.analyze(
                error=error,
                site=site,
                site_config=site_config,
                html_content=html_content,
                run_context=run_context,
            )
            
            # GenerateResult を Dict に変換
            try:
                parsed_analysis = json.loads(analysis_result.text)
                return {
                    "summary": parsed_analysis.get("diagnosis", "分析を実行しました"),
                    "root_causes": [parsed_analysis.get("root_cause", "原因を特定できませんでした")],
                    "suggested_fixes": [parsed_analysis.get("recommended_action", "設定を確認してください")],
                    "confidence": 0.7,  # デフォルト信頼度（将来は LLM の出力から推定）
                    "raw_analysis": parsed_analysis,
                }
            except json.JSONDecodeError:
                # JSON パースに失敗した場合、テキストをそのまま返す
                return {
                    "summary": analysis_result.text[:200],  # 最初の200文字
                    "root_causes": [],
                    "suggested_fixes": [],
                    "confidence": 0.3,
                    "raw_analysis": {"text": analysis_result.text},
                }
        except Exception as e:
            logger.error(f"FailureAnalysisAgent.analyze_failure_context failed: {e}", exc_info=True)
            return {
                "summary": f"分析処理中にエラーが発生しました: {str(e)}",
                "root_causes": [],
                "suggested_fixes": [],
                "confidence": 0.0,
                "error": str(e),
            }

    def _build_prompt(
        self, *, error: Exception, site: str, html_snippet: str, run_context: RunContext, site_config: Dict[str, Any]
    ) -> str:
        """根本原因分析のための詳細なプロンプトを生成する。"""
        error_type = type(error).__name__
        error_message = str(error)
        traceback_info = "".join(traceback.format_tb(error.__traceback__))
        discovery_settings = site_config.get("discovery_settings", {})

        return f"""
# 命令
あなたは、Web自動化システム専門の超一流デバッグ・アナリストです。
実行エージェントがウェブサイト「{site}」の偵察任務中に予期せぬエラーで失敗しました。
提供された以下の状況証拠を総合的に分析し、失敗の根本原因を特定し、具体的な解決策を提案してください。

# 状況証拠
1.  **作戦記録ID (Run ID)**: `{run_context.run_id}`
    - **重要:** この作戦に関する完全な証跡（Trace, HAR, Logs, Screenshots）は、`instance/runs/{run_context.run_id}` ディレクトリに格納されています。

2.  **エラー種別**: `{error_type}`
3.  **エラーメッセージ**: `{error_message}`
4.  **実行時の主要設定**:
    ```json
    {json.dumps(discovery_settings, indent=2, ensure_ascii=False)}
    ```
5.  **発生箇所の詳細 (トレースバック)**:
    ```
    {traceback_info}
    ```
6.  **失敗時点でのHTMLソースコード (冒頭8000文字)**:
    ```html
    {html_snippet}
    ```

# 分析と提案の形式
以下のJSON形式に厳密に従って、日本語で回答してください。思考プロセスは出力に含めないでください。

```json
{{
  "diagnosis": "（ここに、何が起きたのかを簡潔に記述）",
  "root_cause": "（ここに、なぜそうなったのかの根本原因を推測）",
  "recommended_action": "（ここに、開発者が次に何をすべきかの具体的な解決策を提示）"
}}

"""
