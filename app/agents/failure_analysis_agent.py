# 2025年09月21日 15:22 JST
# ファイル名: failure_analysis_agent.py
# レジストリ: app/agents/failure_analysis_agent.py
# バージョン: 9.0.0J (FKB Integration)
#
# --- v9.0.0Jでの主な変更点 ---
# - [FKB統合] Failure Knowledge Base (fkb_local.json) を参照して既知のエラーケースを解決
# - [LLM Controller統合] スタブから本番の AILlmController に切り替え
# - [フォレンジックレポート] 分析結果と証拠リンクを統合して保存

from __future__ import annotations
import logging
import traceback
import json
from typing import Optional, Dict, Any, List
import sys
from pathlib import Path

# --- プロジェクトルートをPythonの検索パスに追加 ---
APP_ROOT = Path(__file__).resolve().parents[2]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

# --- プロジェクトのモジュールをインポート ---
try:
    from app.core.run_context import RunContext
    from app.utils.ai_llm_controller import AILlmController
    from app.models.result_models import GenerateResult
except ImportError as e:
    logging.warning(f"[FailureAnalysisAgent] Import failed, using stubs: {e}")

    class GenerateResult:
        def __init__(self, text: str, cost_usd: float = 0.0):
            self.text = text
            self.cost_usd = cost_usd

    class AILlmController:
        def generate(self, prompt: str, task_type: str) -> GenerateResult:
            logging.info("AILlmController (Stub Fallback): AIへのリクエストをシミュレートします。")
            dummy_response = {
                "diagnosis": "ナビゲーション中にタイムアウトが発生しました。",
                "root_cause": "ターゲットサイトの応答が遅いか、表示を待機すべき特定の要素（セレクタ）がDOM上に存在しなかった可能性があります。",
                "recommended_action": "config/overrides.local.jsonの該当サイト設定で、`timeout_sec`を延長するか、`wait_for_selectors`の値を見直してください。"
            }
            return GenerateResult(text=json.dumps(dummy_response, ensure_ascii=False, indent=2))

logger = logging.getLogger(__name__)


class FKB:
    """Failure Knowledge Base - 失敗パターンのデータベース"""

    def __init__(self, fkb_path: Optional[Path] = None):
        self.fkb_path = fkb_path or APP_ROOT / "config" / "fkb" / "fkb_local.json"
        self.entries: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        """FKBファイルを読み込む"""
        if not self.fkb_path.exists():
            logger.warning(f"[FKB] FKBファイルが見つかりません: {self.fkb_path}")
            return

        try:
            with open(self.fkb_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.entries = data.get("entries", [])
                logger.info(f"[FKB] {len(self.entries)} 件のエントリを読み込みました (v{data.get('version', '?')})")
        except Exception as e:
            logger.error(f"[FKB] FKB読み込みエラー: {e}")

    def find_matching_entry(self, error_message: str, site: str, url: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """エラーシグネチャに一致するエントリを検索"""
        checked = 0
        matched = 0

        for entry in self.entries:
            entry_site = entry.get("site", "")
            # サイト一致チェック
            if entry_site != site and entry_site != "GENERIC":
                checked += 1
                continue

            error_sig = entry.get("error_signature", "")
            # エラーシグネチャ一致チェック
            if error_sig.lower() in error_message.lower():
                logger.info(f"[FKB] ✅ シグネチャマッチ: {entry.get('id')} - '{error_sig}'")
                logger.debug(f"[FKB]   error_message: {error_message[:100]}...")
                matched += 1
                return entry

            # 条件チェック
            conditions = entry.get("conditions", {})
            if url and "final_url_contains" in conditions:
                patterns = conditions["final_url_contains"]
                if any(pattern in url for pattern in patterns):
                    logger.info(f"[FKB] ✅ URL条件マッチ: {entry.get('id')} - patterns: {patterns}")
                    logger.debug(f"[FKB]   url: {url}")
                    matched += 1
                    return entry

            checked += 1

        if checked > 0:
            logger.debug(f"[FKB] チェックしたエントリ: {checked}, マッチ: {matched}")
        return None

    def get_solution(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """エントリから解決策を取得"""
        solution = {
            "id": entry.get("id"),
            "solution_pattern": entry.get("solution_pattern", []),
            "recovery_url": entry.get("recovery_url"),
            "alternate_selectors": entry.get("alternate_selectors", []),
            "root_cause": entry.get("root_cause"),
            "auto_apply_rules": entry.get("auto_apply_rules", {})
        }
        logger.info(f"[FKB] 解決策を取得: {entry.get('id')} - {solution.get('solution_pattern', [])[:2]}")
        return solution


class FailureAnalysisAgent:
    """エージェントの失敗を分析し、原因と対策を提案するメタエージェント。"""

    def __init__(self, runtime_kwargs: Optional[Dict[str, Any]] = None):
        self.runtime_kwargs = runtime_kwargs or {}
        self.llm_controller = AILlmController()
        self.fkb = FKB()

    def analyze(
        self,
        *,
        error: Exception,
        site: str,
        site_config: Dict[str, Any],
        html_content: Optional[str],
        run_context: RunContext,
        current_url: Optional[str] = None
    ) -> GenerateResult:
        """与分析し、AIの分析結果を返す"""
        error_message = str(error)

        # --- FKB チェック ---
        fkb_entry = self.fkb.find_matching_entry(error_message, site, current_url)
        if fkb_entry:
            logger.info(f"[FailureAnalysisAgent] FKBから解決策を生成: {fkb_entry.get('id')}")
            solution = self.fkb.get_solution(fkb_entry)

            forensic_report = {
                "ai_analysis": {
                    "diagnosis": f"既知のエラー: {fkb_entry.get('error_signature')}",
                    "root_cause": solution.get("root_cause"),
                    "recommended_action": f"FKB ID: {solution.get('id')} - 解決策: {', '.join(solution.get('solution_pattern', []))}",
                    "fkb_based": True
                },
                "fkb_entry": fkb_entry,
                "solution": solution,
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
            response_text = json.dumps(forensic_report["ai_analysis"], ensure_ascii=False, indent=2)
            return GenerateResult(text=response_text)

        # --- LLM 分析 ---
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

            parsed_analysis = json.loads(analysis_result.text)

            forensic_report = {
                "ai_analysis": parsed_analysis,
                "fkb_based": False,
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

    def _build_prompt(
        self,
        *,
        error: Exception,
        site: str,
        html_snippet: str,
        run_context: RunContext,
        site_config: Dict[str, Any]
    ) -> str:
        """根本原因分析のための詳細なプロンプトを生成する"""
        error_type = type(error).__name__
        error_message = str(error)
        traceback_info = "".join(traceback.format_tb(error.__traceback__))
        discovery_settings = site_config.get("discovery_settings", {})
        fkb_info = f"FKBには {len(self.fkb.entries)} 件の既知パターンが登録されています。"

        return f"""
# 命令
あなたは、Web自動化システム専門の超一流デバッグ・アナリストです。
実行エージェントがウェブサイト「{site}」の偵察任務中に予期せぬエラーで失敗しました。
提供された以下の状況証拠を総合的に分析し、失敗の根本原因を特定し、具体的な解決策を提案してください。

{fkb_info}

# 状況証拠
1. **作戦記録ID (Run ID)**: `{run_context.run_id}`
2. **エラー種別**: `{error_type}`
3. **エラーメッセージ**: `{error_message}`
4. **実行時の主要設定**:
    ```json
    {json.dumps(discovery_settings, indent=2, ensure_ascii=False)}
    ```
5. **発生箇所の詳細 (トレースバック)**:
    ```
    {traceback_info}
    ```
6. **失敗時点でのHTMLソースコード (冒頭8000文字)**:
    ```html
    {html_snippet}
    ```

# 分析と提案の形式
以下のJSON形式に厳密に従って、日本語で回答してください。

```json
{{
  "diagnosis": "（ここに、何が起きたのかを簡潔に記述）",
  "root_cause": "（ここに、なぜそうなったのかの根本原因を推測）",
  "recommended_action": "（ここに、開発者が次に何をすべきかの具体的な解決策を提示）"
}}
```
"""
