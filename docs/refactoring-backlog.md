# atelier-kyo-manager リファクタリングバックログ

> 最終更新: 2026-05-10
> 前提: 216ファイル評価のコードレビュー結果（2026-05-09/10）に基づく
> 基準: P0 = 影響大/工数小、P1 = 影響大/工数中、P2 = 影響中/工数中

---

## 優先度P0（すぐやる）

### P0-1. レガシー `pricing_calculator.py` の削除 ✅完了（2026-05-10）
- **対象**: `app/utils/pricing_calculator.py`（79行）、`tests/test_pricing_calculator.py`、`tests/test_pricing_calculator_coverage.py`、`tests/test_utils_comprehensive.py` 内の該当import
- **問題概要**: 旧 `pricing_calculator.py` は手数料率 7.3%（`BUYMA_COMMISSION_RATE = 0.073`）を定義しているが、正しい `core/pricing/` モジュールは `constants.py` で 7.7% を定義している。誰も旧モジュールをimportしていない（テストファイルのみ参照）ため、残存は混乱の元。
- **修正方針**: 旧ファイルを削除し、テストファイル内のimport・テストケースを `core/pricing/` に一本化
- **推定工数**: 1時間

### P0-2. `AutoOrderService` のクラスレベル可変デフォルト引数バグ修正 ✅完了（2026-05-10）
- **対象**: `app/services/auto_order_service.py` L69
- **問題概要**: `logs: list[AutoOrderLog] = []` がクラス属性として定義されており、Pythonの可変デフォルト引数バグにより、全インスタンスでリストが共有される
- **修正方針**: `logs` をインスタンス属性に移動（`__init__` 内で `self.logs: list[AutoOrderLog] = []` に初期化）。必要に応じて `dataclass` への移行も検討
- **推定工数**: 0.5時間

### P0-3. `browser_use_agent.py` と `navigation_driver.py` 間の重複関数除去 ✅完了（2026-05-10）
- **対象**: `app/agents/browser_use_agent.py`（L150 `is_same_origin`、L175 `_dedupe_keep_order`）、`app/agents/browser/navigation_driver.py`（L131 `is_same_origin`、L143 `_dedupe_keep_order`）
- **問題概要**: 同一関数が2ファイルにコピペ実装されている（DRY違反）。`browser_use_agent.py` は `navigation_driver.py` をimportしているため、後者側に統一可能
- **修正方針**: `navigation_driver.py` 側を正とし、`browser_use_agent.py` 側は `from app.agents.browser.navigation_driver import is_same_origin, _dedupe_keep_order` に変更
- **推定工数**: 0.5時間

### P0-4. `profitability_agent.py` の過剰 ImportError ガード整理 ✅完了（2026-05-10）
- **対象**: `app/agents/profitability_agent.py`（L32, L53, L65, L76, L83, L149 — 計6箇所）
- **問題概要**: 6つの `except ImportError` ブロックがインラインスタブ付きで定義されている。当ファイルはアプリケーション内部モジュールであり、依存先は常に存在する
- **修正方針**: `try/except ImportError` を通常importに置き換え。`from __future__ import annotations` と `TYPE_CHECKING` を活用して循環参照を回避
- **推定工数**: 1時間

### P0-5. `orchestrator/__init__.py` の過剰 ImportError ガード整理 ✅完了（2026-05-10）
- **対象**: `app/agents/browser/orchestrator/__init__.py`（L29-L70 — 計9箇所）
- **問題概要**: 9つの `try/except ImportError` が連続し、全て `None` フォールバック。アプリ内モジュールへのimportであり、失敗するのはインストール不備のみ
- **修正方針**: 正常importに置き換え。`TYPE_CHECKING` を使って型チェック時のみimport（実行時は遅延import or 直import）
- **推定工数**: 1時間

---

## 優先度P1（次にやる）

### P1-1. ✅ `navigation_driver.py`（4,212行→1,723行）の責務分割
- **対象**: `app/agents/browser/navigation_driver.py`
- **問題概要**: 40メソッドを抱える巨大モノリス。URLバリデーション、トラップ判定、ロケール管理、UI操作（cookie/overlay）、Moncler固有ロジック等が混在
- **実施日**: 2026-05-10
- **結果**: 以下の5モジュールに分割（Mixinパターン + pure function抽出）:
  - ✅ `nav_types.py`（99行）— RejectReason, LinkCandidate, TrapPageDetected, NavigationContext, NavigationOutcome
  - ✅ `url_rules.py`（388行）— URL正規化・バリデーション pure functions
  - ✅ `moncler_nav.py`（322行）— MonclerNavMixin（_collect_moncler_pdp_links 等）
  - ✅ `locale_manager.py`（1,007行）— LocaleMixin（_ensure_expected_locale 等）
  - ✅ `nav_fallbacks.py`（650行）— FallbackMixin（header_search_fallback 等）
  - `navigation_driver.py`（1,723行）— NavigationDriver コア + re-export（後方互換）
- **後方互換**: 7箇所の外部importerは変更不要（re-exportで対応）
- **テスト**: 886 passed, 0 failures

### P1-2. `browser_use_agent.py`（2,597行→1,834行）のスリム化 ✅完了（2026-05-10）
- **対象**: `app/agents/browser_use_agent.py`
- **問題概要**: 38メソッド、2,597行の巨大クラス。P0-3の重複除去に加え、stealth/route/session/deep extraction等のロジックが混在
- **実施日**: 2026-05-10
- **結果**: 以下の5フェーズで763行削減（29%）:
  - ✅ Phase 1: 11メソッドのUI helpers/settings委譲（~200行削減）
  - ✅ Phase 2a: `_normalize_abs_url`, `_looks_like_trap_or_legal`, `_resolve_run_settings` 委譲
  - ✅ Phase 4: 6メソッド/関数のモジュール抽出（~290行削減）
    - `browser/stealth.py`（~120行）— `_setup_init_scripts` 抽出
    - `browser/route_setup.py`（~75行）— `_setup_routes` 抽出
    - `browser/session_config.py`（~100行）— `_build_context_options`, `_get_session_file`, `_apply_saved_session` 抽出
    - `browser/deep_extraction.py`（~170行）— `_run_deep_extraction_phase2` 抽出
    - `visual_regression.py` に `perform_vrt`, `unpack_vrt` 追加
  - Phase 2b/3 は引数差異が大きく委譲困難なため見送り（`_ensure_plp_materialized`, `_collect_pdp_links`, `_plp_header_search_fallback` 等）
- **テスト**: 886 passed, 0 failures

### P1-3. `products.py` ルート（475行）のサービス層抽出
- **対象**: `app/routes/products.py`
- **問題概要**: Flaskルート関数内にビジネスロジックが混在。`import_csv()`（L140-206 ≒ 66行）や `export_csv()`（L207-289 ≒ 82行）等がルートに直接実装
- **修正方針**:
  - CSV入出力ロジックを `app/services/product_csv_service.py`（新規）に抽出
  - パイプライン関連を `PipelineService` に集約（`run_pipeline`, `run_pipeline_batch`）
  - ルートはHTTPリクエスト/レスポンスの処理のみに専念
- **推定工数**: 3-4時間

### P1-4. `sys.path` 操作の集約・除去 ✅完了（2026-05-10）
- **対象**: app/内7モジュール + テスト6ファイル（計14ファイル）。実際は22ファイルに散在していたが、スクリプト9ファイルは正当なエントリポイントとして保持
- **問題概要**: `sys.path.insert(0, APP_ROOT)` が22ファイルに散在。Flask/pytestがパスを設定済みのため、app/内とテストからは不要
- **修正方針**:
  - app/モジュール7ファイル: sys.path + APP_ROOT 除去
  - テスト6ファイル: sys.path 除去（APP_ROOTが他用途なら定義残す）
  - スクリプト9ファイル: 保持（正当なエントリポイント）
  - 併せて5ファイルのフォールバックimport（`from core.xxx`）を直importに統一
- **推定工数**: 2-3時間

### P1-5. `price_intelligence_agent.py` の Selenium → Playwright 移行 ✅完了（2026-05-10）
- **対象**: `app/agents/price_intelligence_agent.py`（295行→270行）、`app/utils/buyma_catalog_manager.py`（452行→430行）
- **問題概要**: プロジェクトのブラウザ自動化は Playwright に統一されているが、この2ファイルのみ Selenium + selenium_stealth を使用。依存ライブラリの増大と保守コストの二重化
- **修正方針**:
  - `price_intelligence_agent.py`: Playwright sync API に書き換え、`selenium`/`selenium_stealth`/`webdriver_manager` 依存を除去 ✅
  - `buyma_catalog_manager.py`: 同様に Playwright sync API に移行 ✅
  - `requirements.txt` から `selenium`, `selenium-stealth`, `webdriver-manager` を削除 → `ai_image_crawler.py` が未移行のため保留
- **推定工数**: 4-6時間（2ファイル合計）
- **残タスク**: `app/utils/ai_image_crawler.py` の Playwright 移行後に requirements.txt から Selenium 除去可能

### P1-6. 残存 ImportError ガードの段階的整理
- **対象**: P0-4/P0-5以外の全ファイル（計25箇所）:
  - `app/utils/ai_llm_controller.py`（4箇所）
  - `app/utils/ai_background_remover.py`（1箇所）
  - `app/utils/ai_image_crawler.py`（1箇所）
  - `app/utils/shipping_agent.py`（1箇所）
  - `app/agents/browser/session_manager.py`（1箇所）
  - `app/agents/browser/navigation_driver.py`（2箇所）
  - `app/agents/selector_repair_agent.py`（2箇所）
  - `app/agents/plugins/base.py`（1箇所）
  - `app/extractors/product_info_extractor.py`（1箇所）
  - `app/core/run_context.py`（2箇所）
  - `app/scripts/run_site.py`（3箇所）
- **問題概要**: アプリ内部モジュールへの `try/except ImportError` が散在し、インラインスタブを多数定義。実行時にimport失敗するのは環境破壊時のみで、スタブ実行は逆にバグを隠蔽
- **修正方針**:
  - 内部モジュールimport → 通常importに変更
  - 外部ライブラリ（`selenium`等）→ `TYPE_CHECKING` 内に移動または requirements.txt で保証
  - `ruff.toml` の `E402` ignore を段階的に除外
- **推定工数**: 3-4時間

---

## 優先度P2（余裕があれば）

### P2-1. `selector_repair_agent.py`（755行）の分割 ✅完了（2026-05-10）
- **対象**: `app/agents/selector_repair_agent.py`
- **問題概要**: 755行の単一ファイル。セレクタ生成・修復・検証の責務が混在
- **実施日**: 2026-05-10
- **結果**: 755行→170行（77%削減）。3モジュールに分割:
  - ✅ `browser/selector_prompt_builder.py`（250行）— `build_selector_repair_prompt`, `extract_site_constraints`, `extract_failed_selector_from_error`, `optimize_dom_snippet`
  - ✅ `browser/selector_ranker.py`（79行）— `rank_selectors`, `matches_site_constraints`, `calculate_specificity`
  - ✅ `browser/selector_validator.py`（52行）— `extract_json_from_text`, `normalize_proposal`
- **バグ修正**: `_build_selector_repair_prompt` の重複 `feedback_section` 2重定義を除去、missing `previous_successes`/`previous_failures` パラメータを追加
- **テスト**: 886 passed, 0 failures

### P2-2. `buyma_catalog_manager.py` の Playwright 移行（P1-5の一部）
- **対象**: `app/utils/buyma_catalog_manager.py`（452行）
- **問題概要**: P1-5に含まれるが、独立した移行タスクとして実施可能
- **修正方針**: Playwright async API へ移行
- **推定工数**: 2-3時間

### P2-3. `ruff.toml` ignore の段階的解消 ✅完了（2026-05-10）
- **対象**: `ruff.toml` の `ignore` セクション
- **実施日**: 2026-05-10
- **結果**:
  - ✅ `SIM117` — 4テストファイルのネスト `with` マージで6件解消、ignoreから除外
  - ✅ `E501` — `ruff format` で22ファイル自動修正。79件残存は長文字列/JSコードのためignore維持
  - 現在のignore: `E501`, `E402`, `F821`（3ルール）

### P2-4. `docs/reports/` のテスト結果ファイル整理 ✅完了（2026-05-10）
- **対象**: `docs/reports/TEST_RESULTS_*.txt`
- **実施日**: 2026-05-10
- **結果**: 既に `.gitignore` に登録済み、実ファイルも未追跡のため作業不要

### P2-5. `plp_driver.py`（1,340行）の分割検討
- **対象**: `app/agents/browser/plp_driver.py`
- **問題概要**: 1,340行の単一ファイル。PLP巡回・ページネーション・抽出等が混在
- **修正方針**: P1-1のnavigation_driver分割完了後に、同様の観点で分割を検討。プラグインパターン（`plugins/base.py` 参考）の適用余地あり
- **推定工数**: 3-4時間

### P2-6. テストカバレッジの底上げ
- **対象**: `tests/` 全体
- **問題概要**: 現在のカバレッジベースラインに対して、agents層・utils層のテストが不十分
- **修正方針**: P0/P1リファクタリング完了後に、分割後モジュールのユニットテストを追加
- **推定工数**: 6-8時間（リファクタリング後の各モジュールに対して）

---

## 完了済み

- [x] Routes大規模リファクタリング（2026-05-09完了）
- [x] services/models/utils import整理（2026-05-09完了）
- [x] 未使用import除去（2026-05-09完了）
- [x] CI/CD品質ゲート導入（2026-05-09完了）
- [x] ruff ignore段階解消 Phase 2-4（2026-05-09完了）

---

## 工数サマリー

| 優先度 | 項目数 | 推定合計工数 |
|--------|--------|-------------|
| P0     | 5      | 4時間       |
| P1     | 6      | 22-30時間   |
| P2     | 6      | 16-22時間   |
| **合計** | **17** | **42-56時間** |

## 依存関係

```
P0-3 (重複除去) → P1-2 (browser_use_agentスリム化) → P2-5 (plp_driver分割)
P0-4/P0-5 (ImportError整理) → P1-6 (残存ImportError整理) → P2-3 (ruff ignore解消)
P1-1 (navigation_driver分割) → P2-5 (plp_driver分割)
P1-5 (Selenium→Playwright) は独立実施可能
P2-6 (テストカバレッジ) は各リファクタリング完了後に実施
```

## 高品質ファイル（変更不要・参考モデル）

| ファイル | 評価 | 参考理由 |
|---------|------|---------|
| `app/core/pricing/calculator.py` | 5/5 | 責務分割・型安全・テスタブル |
| `app/core/pricing/rules.py` | 5/5 | 設定のdataclass化・外部ファイル分離 |
| `app/core/pricing/schemas.py` | 5/5 | 入出力スキーマの明示的定義 |
| `app/agents/plugins/base.py` | 4/5 | プラグインアーキテクチャの基底 |
| `app/agents/moncler/moncler_navigation_policy.py` | 4/5 | ポリシーパターンの適切な実装 |
