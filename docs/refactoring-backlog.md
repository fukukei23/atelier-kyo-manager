# atelier-kyo-manager リファクタリングバックログ

> 最終更新: 2026-05-10
> 前提: 216ファイル評価のコードレビュー結果（2026-05-09/10）に基づく
> 基準: P0 = 影響大/工数小、P1 = 影響大/工数中、P2 = 影響中/工数中

---

## 優先度P0（すぐやる）

### P0-1. レガシー `pricing_calculator.py` の削除
- **対象**: `app/utils/pricing_calculator.py`（79行）、`tests/test_pricing_calculator.py`、`tests/test_pricing_calculator_coverage.py`、`tests/test_utils_comprehensive.py` 内の該当import
- **問題概要**: 旧 `pricing_calculator.py` は手数料率 7.3%（`BUYMA_COMMISSION_RATE = 0.073`）を定義しているが、正しい `core/pricing/` モジュールは `constants.py` で 7.7% を定義している。誰も旧モジュールをimportしていない（テストファイルのみ参照）ため、残存は混乱の元。
- **修正方針**: 旧ファイルを削除し、テストファイル内のimport・テストケースを `core/pricing/` に一本化
- **推定工数**: 1時間

### P0-2. `AutoOrderService` のクラスレベル可変デフォルト引数バグ修正
- **対象**: `app/services/auto_order_service.py` L69
- **問題概要**: `logs: list[AutoOrderLog] = []` がクラス属性として定義されており、Pythonの可変デフォルト引数バグにより、全インスタンスでリストが共有される
- **修正方針**: `logs` をインスタンス属性に移動（`__init__` 内で `self.logs: list[AutoOrderLog] = []` に初期化）。必要に応じて `dataclass` への移行も検討
- **推定工数**: 0.5時間

### P0-3. `browser_use_agent.py` と `navigation_driver.py` 間の重複関数除去
- **対象**: `app/agents/browser_use_agent.py`（L150 `is_same_origin`、L175 `_dedupe_keep_order`）、`app/agents/browser/navigation_driver.py`（L131 `is_same_origin`、L143 `_dedupe_keep_order`）
- **問題概要**: 同一関数が2ファイルにコピペ実装されている（DRY違反）。`browser_use_agent.py` は `navigation_driver.py` をimportしているため、後者側に統一可能
- **修正方針**: `navigation_driver.py` 側を正とし、`browser_use_agent.py` 側は `from app.agents.browser.navigation_driver import is_same_origin, _dedupe_keep_order` に変更
- **推定工数**: 0.5時間

### P0-4. `profitability_agent.py` の過剰 ImportError ガード整理
- **対象**: `app/agents/profitability_agent.py`（L32, L53, L65, L76, L83, L149 — 計6箇所）
- **問題概要**: 6つの `except ImportError` ブロックがインラインスタブ付きで定義されている。当ファイルはアプリケーション内部モジュールであり、依存先は常に存在する
- **修正方針**: `try/except ImportError` を通常importに置き換え。`from __future__ import annotations` と `TYPE_CHECKING` を活用して循環参照を回避
- **推定工数**: 1時間

### P0-5. `orchestrator/__init__.py` の過剰 ImportError ガード整理
- **対象**: `app/agents/browser/orchestrator/__init__.py`（L29-L70 — 計9箇所）
- **問題概要**: 9つの `try/except ImportError` が連続し、全て `None` フォールバック。アプリ内モジュールへのimportであり、失敗するのはインストール不備のみ
- **修正方針**: 正常importに置き換え。`TYPE_CHECKING` を使って型チェック時のみimport（実行時は遅延import or 直import）
- **推定工数**: 1時間

---

## 優先度P1（次にやる）

### P1-1. `navigation_driver.py`（4,212行）の責務分割
- **対象**: `app/agents/browser/navigation_driver.py`
- **問題概要**: 40メソッドを抱える巨大モノリス。URLバリデーション、トラップ判定、ロケール管理、UI操作（cookie/overlay）、Moncler固有ロジック等が混在
- **修正方針**: 以下のように責務ごとに分割:
  - `url_utils.py` — URL正規化・バリデーション（`_normalize_candidate_url`, `_validate_candidate_url`, `_classify_candidate`, `_extract_origin` 等）
  - `trap_detector.py` — トラップページ判定（`_detect_trap_page`, `_looks_like_trap_or_legal`, `is_expected_locale_path` 等）
  - `locale_manager.py` — ロケール管理（`_ensure_expected_locale`, `_is_locale_stable` 等）
  - `moncler_handler.py` — Moncler固有（`_collect_moncler_pdp_links`, `_is_valid_moncler_pdp_url`, `_trigger_moncler_self_healing` 等）
  - `navigation_driver.py` — コアナビゲーションのみ（`run_plp_flow`, `collect_pdp_links`, `NavigationContext`, `NavigationOutcome`, `NavigationDriver` スケルトン）
- **参考**: 既に分割済みの `plugins/base.py`（4/5）、`moncler_navigation_policy.py`（4/5）が良いモデル
- **推定工数**: 6-8時間（テスト修正含む）

### P1-2. `browser_use_agent.py`（2,605行）のスリム化
- **対象**: `app/agents/browser_use_agent.py`
- **問題概要**: 46メソッド、2,605行の巨大クラス。P0-3の重複除去に加え、Monclerパッチやdeep extraction等のロジックが混在
- **修正方針**:
  - P0-3の重複除去（正: `navigation_driver.py` 側）
  - Moncler固有ロジックを `browser_use_moncler_patch.py` または `moncler/` に集約
  - deep extraction（`_run_deep_extraction_phase2` 等）を `extractor.py` に移動
  - 残りのコアフロー（PLP→PDP）は `BrowserOrchestrator` に委譲済みの部分を確認し、重複を排除
- **推定工数**: 4-6時間

### P1-3. `products.py` ルート（475行）のサービス層抽出
- **対象**: `app/routes/products.py`
- **問題概要**: Flaskルート関数内にビジネスロジックが混在。`import_csv()`（L140-206 ≒ 66行）や `export_csv()`（L207-289 ≒ 82行）等がルートに直接実装
- **修正方針**:
  - CSV入出力ロジックを `app/services/product_csv_service.py`（新規）に抽出
  - パイプライン関連を `PipelineService` に集約（`run_pipeline`, `run_pipeline_batch`）
  - ルートはHTTPリクエスト/レスポンスの処理のみに専念
- **推定工数**: 3-4時間

### P1-4. `sys.path` 操作の集約・除去
- **対象**: `app/utils/ai_research_orchestrator.py`、`app/agents/reporting_agent.py`、`app/agents/selector_discovery_agent.py`、`app/scripts/run_site.py`、`app/agents/failure_analysis_agent.py`、`app/agents/self_healing_agent.py`、`app/agents/supplier_scout_agent.py`、`app/agents/page_recovery_agent.py`（計8ファイル）
- **問題概要**: `sys.path.insert(0, APP_ROOT)` が8ファイルに散在。Flaskアプリとして `python -m` 実行やパッケージインストール済みであれば不要
- **修正方針**:
  - `pyproject.toml` または `setup.py` でパッケージをインストール可能にする
  - または単一の `app/__init__.py` で `sys.path` を操作し、各ファイルから除去
  - `scripts/run_site.py` はエントリポイントとして残すが、他ファイルからは除去
- **推定工数**: 2-3時間

### P1-5. `price_intelligence_agent.py` の Selenium → Playwright 移行
- **対象**: `app/agents/price_intelligence_agent.py`（295行、Selenium依存）、`app/utils/buyma_catalog_manager.py`（452行、Selenium依存）
- **問題概要**: プロジェクトのブラウザ自動化は Playwright に統一されているが、この2ファイルのみ Selenium + selenium_stealth を使用。依存ライブラリの増大と保守コストの二重化
- **修正方針**:
  - `price_intelligence_agent.py`: Playwright async API に書き換え、`selenium`/`selenium_stealth`/`webdriver_manager` 依存を除去
  - `buyma_catalog_manager.py`: 同様に Playwright に移行
  - `requirements.txt` から `selenium`, `selenium-stealth`, `webdriver-manager` を削除
- **推定工数**: 4-6時間（2ファイル合計）

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

### P2-1. `selector_repair_agent.py`（755行）の分割
- **対象**: `app/agents/selector_repair_agent.py`
- **問題概要**: 755行の単一ファイル。セレクタ生成・修復・検証の責務が混在
- **修正方針**: 生成ロジックと修復ロジックを分離し、`selector_generator.py` + `selector_repair.py` に分割
- **推定工数**: 2-3時間

### P2-2. `buyma_catalog_manager.py` の Playwright 移行（P1-5の一部）
- **対象**: `app/utils/buyma_catalog_manager.py`（452行）
- **問題概要**: P1-5に含まれるが、独立した移行タスクとして実施可能
- **修正方針**: Playwright async API へ移行
- **推定工数**: 2-3時間

### P2-3. `ruff.toml` ignore の段階的解消（Phase 5）
- **対象**: `ruff.toml` の `ignore` セクション（`E501`, `E402`, `F821`, `SIM117`）
- **問題概要**: Phase 4 まで完了済み。残り4ルールが ignore されている
- **修正方針**:
  - `E402` → P1-6のImportError整理完了後に除外
  - `F821` → forward reference/conditional importの修正後に除外
  - `E501` → `ruff format` で自動整形（`line-length = 120` で既に設定済み）
  - `SIM117` → 段階的に `with` 文をマージ
- **推定工数**: 2-3時間

### P2-4. `docs/reports/` のテスト結果ファイル整理
- **対象**: `docs/reports/TEST_RESULTS_*.txt`（44ファイル）
- **問題概要**: CI/CDのテスト結果ファイルがリポジトリ内に蓄積。gitignoreすべき
- **修正方針**: `.gitignore` に `docs/reports/TEST_RESULTS_*.txt` を追加し、既存ファイルを `git rm`
- **推定工数**: 0.5時間

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
