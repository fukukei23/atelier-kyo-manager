# ==============================================================================
# ファイル名 (File Name): selector_repair_agent.py
# レジストリ (Registry): app/agents/selector_repair_agent.py
# 更新日時 (Date & Time JST): 2025年09月21日 20:55:00
# バージョン (Version): 8.0.0J (Async Fix)
#
# --- v8.0.0Jでの主な変更点 ---
# - [非同期化] 実行ログとの整合性を取るため、`propose_fix`を
#   `async def`として再定義しました。LLMの呼び出しは非同期I/Oを
#   伴うため、これはアーキテクチャとしてより正しい姿です。
# ==============================================================================
# -*- coding: utf-8 -*-
from __future__ import annotations
import logging
import json
from typing import Optional, Dict, Any, List
import sys
from pathlib import Path
import asyncio

# --- プロジェクトルートをPythonの検索パスに追加 ---
APP_ROOT = Path(__file__).resolve().parents[2]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from core.run_context import RunContext
# from app.utils.ai_llm_controller import AILlmController

# --- スタブクラス (開発用) ---
class GenerateResult:
    def __init__(self, text: str, cost_usd: float = 0.0):
        self.text = text
        self.cost_usd = cost_usd

class AILlmController:
    async def generate(self, prompt: str, task_type: str) -> GenerateResult:
        logging.info("AILlmController (Stub): AIへの非同期リクエストをシミュレートします。")
        await asyncio.sleep(1.5) # I/Oバウンドな処理を模倣
        dummy_response = {
          "proposed_selectors": [".product-card__title", "a[data-e2e='product-link']"],
          "rationale": "提供されたHTMLスニペットと失敗したセレクタに基づき、より安定している可能性が高いテストIDまたはBEMスタイルのクラス名を提案します。"
        }
        return GenerateResult(text=json.dumps(dummy_response, ensure_ascii=False, indent=2))
# --- スタブここまで ---

logger = logging.getLogger(__name__)

class SelectorRepairAgent:
    """情報分析官。AIを用いて壊れたセレクタを修復・提案する。"""
    def __init__(self):
        self.llm_controller = AILlmController()

    async def propose_fix(self, *, intent: str, failed_selectors: List[str], html_content: str, site: str, site_config: Dict, run_context: RunContext) -> Dict[str, Any]:
        prompt = self._build_prompt(intent=intent, failed_selectors=failed_selectors, html_snippet=html_content, site=site, run_context=run_context)

        logger.info(f"AIによるセレクタ修復を非同期で要請します (RunID: {run_context.run_id})")
        try:
            result = await self.llm_controller.generate(prompt, task_type="code")
            proposal = json.loads(result.text)
            run_context.save_json("selector_repair_proposal.json", proposal)
            return proposal
        except Exception as e:
            logger.error(f"セレクタ修復のためのLLM呼び出しに失敗: {e}", exc_info=True)
            return {}

    def _build_prompt(self, *, intent: str, failed_selectors: List[str], html_snippet: str, site: str, run_context: RunContext) -> str:
        # (プロンプト構築ロジックは変更なし)
        return f"""
# Role
あなたは、Web自動化とフロントエンド開発を専門とする世界トップクラスのAIエンジニアです。
# Goal
ウェブサイト「{site}」の自動化タスクが、古いCSSセレクタが原因で失敗しました。あなたの任務は、提供されたHTMLの状況証拠を分析し、失敗したタスクの「意図」を達成するための、新しく、より堅牢で、将来の変更に強いCSSセレクタを提案することです。
# Context
- Website: {site}
- Intent: {intent}
- Previously Failed Selectors: {json.dumps(failed_selectors)}
- Run ID: {run_context.run_id}
# Evidence
- HTML Snippet:
  ```html
  {html_snippet[:8000]}
  ```
# Instructions
1. HTMLを分析し、「Intent」に最も合致する要素を特定してください。
2. 最大3つまで、信頼性の高いCSSセレクタを提案してください。
3. 応答は、以下のJSON形式のみで、厳密に出力してください。
```json
{{
  "proposed_selectors": [],
  "rationale": "なぜそのセレクタを提案したかの理論的根拠"
}}
```
"""
