# 完了レポート: LLM統合（GLM統一）

**日付**: 2026-03-22
**担当者**: AI Assistant

---

## 目的

LLMプロバイダをGLM（智譜AI）に統一し、他のLLM（OpenAI、Gemini、MiniMax、DeepSeek等）を削除。

---

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|----------|
| `app/utils/ai_llm_controller.py` | GLMのみに統合、ポリシー/コスト/client更新 |
| `app/config/llm_costs.json` | GLM価格のみ残置 |
| `app/config/config.py` | `GLM_API_KEY`のみ保持 |
| `app/config/sites/overrides.local.json` | LLM順序をGLMのみに |
| `.env` | 全LLMキーを削除、GLMキーのみ追加 |
| `app/utils/interactive_repair_session.py` | コメント更新 |
| `app/utils/ai_generate_descriptions.py` | コメント更新 |
| `app/agents/profitability_agent.py` | コメント更新 |
| `docs/GLM_SETUP.md` | **新規作成** - GLM設定ガイド |

---

## 設定内容

**.env設定**:
```bash
GLM_API_KEY=your-glm-api-key-here
GLM_MODEL=glm-4
GLM_API_BASE=https://open.bigmodel.cn/api/paas/v4
```

**フォールバック順序**: `glm` → `local` (Llama)

---

## 次のステップ

1. `GLM_API_KEY`を.envに設定
2. 動作確認テスト実行
