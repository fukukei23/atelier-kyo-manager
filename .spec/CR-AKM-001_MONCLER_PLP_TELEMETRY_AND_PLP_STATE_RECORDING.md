# CR-AKM-001: Moncler PLP Telemetry & PLP State Recording

- **Status:** In-Progress

- **Author:** [よしひろ]

- **Date:** 2025-12-04

- **Related Issues / PRs:** 

  - docs/moncler/PHASE1_STATUS_SUMMARY.md

  - docs/moncler/PLP_EXTRACTION_FIX_TASK_TEMPLATE.md

  - docs/moncler/PHASE1_5_DRY_RUN_REPORT.md

## 1. Overview & Context

- **目的 (Why):**  

  Moncler PLP 実行時に、`plp_dom_initial_materialized.html` と `selector_counts_plp_initial.json` が生成されず、  

  失敗時の DOM/セレクタ状態を十分に観測できていない。  

  Phase 1 のゴールは「壊れ方を観測できる状態」を作ることなので、  

  Telemetry 統合と PLP state recording を仕上げる。

- **背景 (Background):**

  - `NavigationDriver` 内で PLP 状態を記録するフックを追加済み。

  - しかし実行ログ上では、以下のような警告が出ている：

    - `Telemetry object (TelemetryClient/TelemetryService) does not have expected interface`

    - `Available attributes: ['save_dom', 'save_json', 'save_screenshot', 'write_fail_snapshot']`

    - `Telemetry service not available, skipping PLP state recording`

  - その結果、`tiles_detected=True` に到達しても、run ディレクトリ直下に

    - `plp_dom_initial_materialized.html`

    - `selector_counts_plp_initial.json`

    が生成されていない。

  - Moncler PLP 自体も、`/en-lt/en-int/search` に飛ばされ PDP リンク 0 件となっており、

    セレクタ調整より前に「失敗時のスナップショット」を安定して残せる状態が必要。

- **参照 (References):**

  - docs/moncler/PHASE1_STATUS_SUMMARY.md

  - docs/moncler/PLP_EXTRACTION_FIX_TASK_TEMPLATE.md

  - docs/moncler/PHASE1_5_DRY_RUN_REPORT.md

  - run logs (`instance/runs/20251204_162621_157/system.log`)

## 2. Scope

### ✅ In-Scope (やること)

- [x] TelemetryClient / TelemetryService に共通メソッド `record_plp_state(...)` を追加する。

- [x] `NavigationDriver` から Telemetry の `record_plp_state` を呼び出すよう統一する。

- [ ] Moncler 実行時に、以下のファイルを必ず出力する：

  - `instance/runs/<run_id>/plp_dom_initial_materialized.html`

  - `instance/runs/<run_id>/selector_counts_plp_initial.json`

- [ ] 上記の動作を確認したうえで、簡易レポートを docs/moncler/ に追加する  

      (例: `PLP_TELEMETRY_ADAPTER_COMPLETION_REPORT.md`)。

### ❌ Out-of-Scope (やらないこと)

- [ ] Moncler の PLP→PDP 抽出ロジックそのものの修正（URL パターン・セレクタ調整）。  

      → これは別 CR（CR-AKM-002予定）で扱う。

- [ ] 他サイト（Moncler 以外）の Telemetry 拡張。

- [ ] PLP materialization ロジック（リトライ戦略・タイムアウト・Stealth）の再設計。  

      → Phase 1 ですでに完了済みのため対象外。

## 3. Implementation Plan

1. **既存 Telemetry 実装の調査** ✅

   - TelemetryClient / TelemetryService の定義ファイルを特定。

   - 既存メソッド (`save_dom`, `save_json`, `save_screenshot`, `write_fail_snapshot`) のインタフェースを確認。

   - `NavigationDriver` から現在どのように呼ばれているかを確認。

2. **共通 API `record_plp_state` の設計** ✅

   - 想定シグネチャ例：

     ```python

     async def record_plp_state(

         self,

         page: Any,

         *,

         name: str = "plp_dom_initial_materialized",

         selectors: Optional[List[str]] = None,

         site_config: Optional[Dict[str, Any]] = None,

     ) -> None:

         ...

     ```

   - 内部で `save_dom` / `save_json` を利用し、ファイル名を固定：

     - DOM: `plp_dom_initial_materialized.html`

     - セレクタカウント: `selector_counts_plp_initial.json`

3. **TelemetryClient / TelemetryService への実装** ✅

   - 両クラスに `record_plp_state` を追加し、同じインタフェースを提供。

   - 例外発生時には logger.warn でログを残し、呼び出し元に例外を伝播させない。

4. **NavigationDriver の呼び出し側修正** ✅

   - `hasattr(telemetry, "record_plp_state")` 等のインタフェースチェックを簡素化。

   - Telemetry が `None` でない場合は、単純に `await telemetry.record_plp_state(...)` を呼ぶ実装に統一。

   - 「Telemetry object does not have expected interface」系の警告ログを削除。

5. **動作確認** 🔄

   - `python -m app.scripts.run_site moncler --query "down jacket" --headful`

     を実行し、最新 `run_id` 配下に以下が生成されることを確認：

     - `plp_dom_initial_materialized.html`

     - `selector_counts_plp_initial.json`

   - system.log に Telemetry まわりの WARNING が出ていないことを確認。

6. **レポート作成** ⏳

   - `docs/moncler/PLP_TELEMETRY_ADAPTER_COMPLETION_REPORT.md` を作成：

     - 修正ファイル一覧

     - Before / After のログ比較

     - 生成されたファイルのサンプル

     - 今後（CR-AKM-002: PLP selector / URL validation 修正）への接続点

## 4. Testing Strategy

- **テスト方針:**

  - E2E に近い「実行テスト」を中心にする。

    - Moncler 向け run_site を実行し、実際に run ディレクトリ配下のファイル生成を確認。

  - Telemetry 単体レベルでは、可能なら簡易ユニットテストを追加：

    - ダミーの `html` / `selector_counts` を渡し、所定のパスにファイルが作られることを検証。

- **テストコマンド:**

  - 手動実行：

    - `python -m app.scripts.run_site moncler --query "down jacket" --headful`

  - （任意）ユニットテスト:

    - `pytest tests/telemetry/test_record_plp_state.py -q`

- **主な検証観点:**

  - 正常系：

    - tiles_detected=True となった run で、PLP スナップショットファイルが必ず生成される。

  - 異常系：

    - Telemetry が None の場合、NavigationDriver はエラーにならずに処理を継続する。

    - Telemetry 内部で例外が出ても、run 全体がクラッシュしない（ログに WARN が残るだけ）。

  - 既存機能への影響：

    - `save_dom` / `save_json` / `write_fail_snapshot` を利用している既存コードに影響がないこと。

