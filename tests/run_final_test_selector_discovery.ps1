# ==============================================================================
# File: run_final_test_selector_discovery.ps1
# Registry: C:\Users\USER\tools\atelier-kyo-manager\tests\run_final_test_selector_discovery.ps1
# Date & Time (JST): 2025-09-14 13:00:00
# Version: 2.0 (Final)
#
# --- 事前準備 ---
# `app/config/sites/overrides.local.json` ファイルは、正常な状態に
# 戻しておいてください。（AIがサイトを正しくナビゲートできる必要があるため）
#
# --- 使用方法 ---
# 1. PowerShellターミナルで、プロジェクトのルートディレクトリから以下を実行します。
#    > .\run_final_test_selector_discovery.ps1
# ==============================================================================

# AIセレクタ発見（地図作成官の出動テスト）
# SSENSEサイトに対して、セレクタ発見任務を命令します。
python -m app.utils.ai_research_orchestrator "Gucci" --suppliers SSENSE --discover-selectors
