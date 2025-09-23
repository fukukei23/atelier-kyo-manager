# ==============================================================================
# File: run_final_test_failure_analysis.ps1
# Registry: C:\Users\USER\tools\atelier-kyo-manager\tests\run_final_test_failure_analysis.ps1
# Date & Time (JST): 2025-09-14 13:00:00
# Version: 2.0 (Final)
#
# --- 事前準備 ---
# 1. このテストは、AI事故調査官が正しく出動するかを確認するものです。
#    ウェブ調査が【必ず失敗する】状況を意図的に作る必要があります。
# 2. `app/config/sites/overrides.local.json` ファイルを開き、
#    "SSENSE" のセレクタを、以下のように意図的に無効なものに書き換えてください。
#
#    {
#      "SSENSE": {
#        "search_input_selectors": ["input#this-is-an-invalid-selector"]
#      }
#    }
#
# --- 使用方法 ---
# 1. 上記の事前準備が完了したら、PowerShellターミナルで以下を実行します。
#    > .\run_final_test_failure_analysis.ps1
# ==============================================================================

# AI失敗分析（事故調査官の出動テスト）
# Pythonの `-m` オプションを使い、`ModuleNotFoundError`を解決した正しい方法で実行します。
python -m app.utils.ai_research_orchestrator "Gucci" --suppliers SSENSE
