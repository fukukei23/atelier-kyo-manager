# テストレポート - 2026年03月21日

## テスト実行結果

```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.0.1, pluggy-1.6.0
rootdir: /mnt/c/Users/USER/tools/atelier-kyo-manager
collected 14 items

tests/test_app_smoke.py::SmokeTestApp::test_app_factory_creates_flask_app PASSED [  7%]
tests/test_app_smoke.py::SmokeTestApp::test_app_has_blueprint PASSED     [ 14%]
tests/test_e2e_integration.py::test_basic_execution SKIPPED (E2E tes...) [ 21%]
tests/test_e2e_integration.py::test_proxy_execution SKIPPED (E2E tes...) [ 28%]
tests/test_e2e_integration.py::test_auto_heal SKIPPED (E2E tests req...) [ 35%]
tests/test_e2e_integration.py::test_video_recording SKIPPED (E2E tests req...) [ 42%]
tests/test_llm_controller.py::test_deepseek_connection SKIPPED (No L...) [ 50%]
tests/test_rembg.py::test_rembg_background_removal SKIPPED (テストデ...) [ 57%]
tests/test_selector_repair_agent.py::test_propose_fix_with_mock_llm PASSED [ 64%]
tests/test_selector_repair_agent.py::test_propose_fix_with_fallback_when_llm_unavailable PASSED [ 71%]
tests/test_selector_repair_agent.py::test_propose_fix_handles_json_parse_error PASSED [ 78%]
tests/test_selector_repair_agent.py::test_propose_fix_with_empty_failed_selectors PASSED [ 85%]
tests/test_selector_repair_agent.py::test_propose_extracts_selectors_from_response PASSED [ 92%]
tests/test_session_manager.py::test_session_manager_open_and_close PASSED [100%]

======================== 8 passed, 6 skipped in 41.37s =========================
```

## サマリー

| 項目 | 値 |
|------|-----|
| 合計テスト数 | 14 |
| 成功 | 8 |
| スキップ | 6 |
| 失敗 | 0 |
| 成功率 | 100% (成功÷実行) |
| 実行時間 | 41.37秒 |

## スキップ内訳

| テスト | 理由 |
|--------|------|
| `test_basic_execution` | E2E tests require Windows environment with subprocess |
| `test_proxy_execution` | E2E tests require Windows environment with subprocess |
| `test_auto_heal` | E2E tests require Windows environment with subprocess |
| `test_video_recording` | E2E tests require Windows environment with subprocess |
| `test_deepseek_connection` | No LLM client configured (API keys required) |
| `test_rembg_background_removal` | テストデータが不足 |

## 今回追加したテスト

### `test_selector_repair_agent.py` (5テスト新規追加)

| テスト名 | 内容 |
|---------|------|
| `test_propose_fix_with_mock_llm` | モックLLMを使ってpropose_fixをテスト |
| `test_propose_fix_with_fallback_when_llm_unavailable` | LLM利用不可時のフォールバックテスト |
| `test_propose_fix_handles_json_parse_error` | JSON解析エラー処理のテスト |
| `test_propose_fix_with_empty_failed_selectors` | 空リスト入力のテスト |
| `test_propose_extracts_selectors_from_response` | 応答からのセレクタ抽出テスト |

## 修正した問題

1. **`test_e2e_integration.py` - `is_wsl()` バグ**
   - `platform.uname().lower()` → `platform.uname().release.lower()`

2. **`test_llm_controller.py` - APIクライアント未設定時のスキップ**
   - クライアント可用性チェックでスキップ

3. **`ai_llm_controller.py` - resultがNoneのケース対応**
   - NoneチェックとValueError追加

4. **`test_selector_repair_agent.py` - MockRunContextの不足メソッド**
   - `save_json` メソッドを追加
