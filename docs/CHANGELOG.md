# Changelog

> プロジェクト固有の変更履歴。SSOT（`obsidian-ssot/01_DECISIONS/`）と併せて参照すること。

## 2026-05-10

### Selenium → Playwright 移行 (P1-5)
- `price_intelligence_agent.py`（v14.0.0J）と `buyma_catalog_manager.py`（v2.0.0）を Selenium から Playwright sync API に全面書き換え
- selenium/selenium-stealth/webdriver-manager 依存を除去（`ai_image_crawler.py` は別タスクで残存）
- テスト 911 passed、回帰なし

### P0リファクタリングバックログ一括実行 (P0-1〜P0-5)

#### P0-1: レガシー pricing_calculator.py 削除
- 旧 `app/utils/pricing_calculator.py`（手数料率 7.3%、誤）とテスト3ファイルを削除
- 正しい 7.7% は `app/core/pricing/constants.py` に定義済み
- 削除ファイル: `app/utils/pricing_calculator.py`, `tests/test_pricing_calculator.py`, `tests/test_pricing_calculator_coverage.py`
- `tests/test_utils_comprehensive.py` から該当 import/テスト除去

#### P0-2: AutoOrderService 可変デフォルト引数バグ修正
- `app/services/auto_order_service.py`: `logs: list = []` をクラス属性→`__init__` 内インスタンス属性に移動
- 全インスタンス間でリストが共有されるバグを修正

#### P0-3: browser_use_agent.py 重複関数除去
- `is_same_origin` と `_dedupe_keep_order` のコピペ実装を `navigation_driver.py` に統一
- `browser_use_agent.py` は `from app.agents.browser.navigation_driver import ...` に変更

#### P0-4: profitability_agent.py ImportError ガード整理
- 6箇所の `try/except ImportError` を直接 import に置換
- `LLM_AVAILABLE`/`SHIPPING_AGENT_AVAILABLE`/`PRICING_CONFIG_AVAILABLE` フラグ除去
- デッドコード除去: `_generate_assessment_summary` の到達不能 return、`_get_dynamic_shipping_cost` の重複 try/except
- テスト修正: `test_profitability_agent.py` 15テスト全面書き直し

#### P0-5: orchestrator/__init__.py ImportError ガード整理
- 9箇所中7箇所を直接 import に変更
- 2箇所（`e2e_success_stage`/`self_healing_policy`）は未実装モジュールのため正当な optional import として保持
- `__init__` の if/elif/else パターンを `or` パターンに簡素化

### テスト結果
- **886 passed, 0 failed, 6 skipped**（旧911から旧テストファイル削除分 -25、機能退化なし）
