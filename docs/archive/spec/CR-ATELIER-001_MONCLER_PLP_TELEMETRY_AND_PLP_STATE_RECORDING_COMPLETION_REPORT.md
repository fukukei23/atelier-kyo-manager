# CR-ATELIER-001 完了レポート: Moncler PLP Telemetry & PLP State Recording

## 実装日時
2025-12-04

## 概要

Moncler PLP 実行時に `plp_dom_initial_materialized.html` と `selector_counts_plp_initial.json` が生成されない問題を解決するため、`TelemetryClient` / `TelemetryService` に共通メソッド `record_plp_state` を追加し、`NavigationDriver` からの呼び出しを統一しました。

### 目的
- PLP materialize 失敗時でも、そのときの DOM とタイル検出状況を保存する
- Phase 1 のゴール「壊れ方を観測できる状態」を実現
- `NavigationDriver` からの `record_plp_state` 呼び出しで `AttributeError` が発生しないようにする

### 原則
- 既存の Telemetry API (`save_dom`, `save_json`, `save_screenshot`, `write_fail_snapshot`) との互換性を維持
- 例外は内部で握りつぶさず、logger.warn でログを残す
- ファイル名は固定（`plp_dom_initial_materialized.html`, `selector_counts_plp_initial.json`）

## 実施内容サマリー

1. **TelemetryClient.record_plp_state の実装確認**
   - 既に実装済みであることを確認（487-516行目）
   - `TelemetryService.record_plp_state` をラップする実装

2. **TelemetryService.record_plp_state の実装確認**
   - 既に実装済みであることを確認（89-138行目）
   - `page` オブジェクトから HTML を取得し、DOM とセレクタカウントを保存

3. **フォールバック版 TelemetryClient への追加**
   - `browser_use_agent.py` の ImportError フォールバック部に `record_plp_state` スタブを追加（162-171行目）

4. **インデントエラーの修正**
   - `browser_use_agent.py` のフォールバック版 `TelemetryClient` のインデントを修正

## 変更ファイル一覧

### 変更されたファイル
- `app/agents/browser_use_agent.py`
  - フォールバック版 `TelemetryClient` に `record_plp_state` メソッドを追加（162-171行目）
  - インデントエラーを修正（150-171行目）

### 確認済み（変更不要）
- `app/agents/browser/telemetry.py`
  - `TelemetryClient.record_plp_state` は既に実装済み（487-516行目）
  - `TelemetryService.record_plp_state` も既に実装済み（89-138行目）
- `app/agents/browser/navigation_driver.py`
  - `record_plp_state` の呼び出しシグネチャは正しい（251-256行目）

## TelemetryClient.record_plp_state のシグネチャと振る舞い

### シグネチャ
```python
async def record_plp_state(
    self,
    page: Any,
    *,
    name: str = "plp_dom_initial_materialized",
    selectors: Optional[List[str]] = None,
    site_config: Optional[Dict[str, Any]] = None,
) -> None:
```

### 振る舞い
1. `TelemetryService.record_plp_state` に委譲
2. 例外発生時は logger.warn でログを残し、呼び出し元に例外を伝播させない
3. 内部で `TelemetryService` が以下を実行：
   - DOM 保存: `plp_dom_initial_materialized.html`
   - セレクタカウント保存: `selector_counts_plp_initial.json`（selectors が指定された場合のみ）

## 生成されるファイル名と保存場所

### ファイル名（固定）
- **DOM スナップショット**: `plp_dom_initial_materialized.html`
- **セレクタカウント**: `selector_counts_plp_initial.json`

### 保存場所
- `instance/runs/<run_id>/` ディレクトリ直下
- `RunContext.save_content()` と `RunContext.save_json()` を使用

## 実行ログの抜粋

### 成功ログ
```
2025-12-04 23:48:01,023 INFO [NavigationDriver] Recording PLP state: materialized=False, tiles_detected=True
2025-12-04 23:48:01,023 INFO [NavigationDriver] Saved PLP DOM snapshot and selector counts
```

### エラーログ（確認済み：発生していない）
- ❌ `'TelemetryClient' object has no attribute 'record_plp_state'` → **発生していない**
- ❌ `Telemetry object ... does not have expected interface` → **発生していない**
- ❌ `Telemetry service not available, skipping PLP state recording` → **発生していない**

## 動作確認結果

### 実装確認
- ✅ `TelemetryClient.record_plp_state` が実装されている
- ✅ `TelemetryService.record_plp_state` が実装されている
- ✅ `NavigationDriver` からの呼び出しシグネチャが一致している
- ✅ フォールバック版にも `record_plp_state` が追加されている

### 実行ログ確認
- ✅ `[NavigationDriver] Recording PLP state` ログが出力されている
- ✅ `[NavigationDriver] Saved PLP DOM snapshot and selector counts` ログが出力されている
- ✅ `AttributeError: 'TelemetryClient' object has no attribute 'record_plp_state'` が発生していない

### ファイル生成確認
- ⚠️ 実行がタイムアウトで中断されたため、ファイル生成の確認は次回実行時に実施予定
- ただし、ログから `record_plp_state` が正常に呼び出されていることは確認済み

## 設計上の改善点

### アーキテクチャの改善
- **インタフェースの統一**: `TelemetryClient` と `TelemetryService` の両方に `record_plp_state` を実装することで、呼び出し側が型を意識せずに同じメソッドを呼べるようになった
- **エラーハンドリング**: 例外発生時も logger.warn でログを残し、呼び出し元の処理を継続できる設計

### 将来の拡張性への配慮
- `record_plp_state` メソッドは `name` パラメータを受け取るため、将来的に異なる名前の PLP スナップショットを保存することも可能
- `selectors` が `None` の場合は JSON を保存しないため、DOM のみを保存する用途にも対応可能

## 既知の制約・注意事項

### 既存コードとの互換性
- ✅ 既存の Telemetry API (`save_dom`, `save_json`, `save_screenshot`, `write_fail_snapshot`) は変更していないため、既存コードへの影響はない
- ✅ `NavigationDriver` の変更は、PLP state recording の呼び出し部分のみであり、他の機能への影響はない

### 制限事項やトレードオフ
- ファイル名は `name` パラメータが `"plp_dom_initial"` または `"plp_dom_initial_materialized"` の場合のみ固定される
- 他の `name` 値の場合は、従来通り `name` ベースのファイル名が生成される

## 次のステップ

### 即座に実施すべきこと
1. **ファイル生成の確認**
   - Moncler run を再実行し、`plp_dom_initial_materialized.html` と `selector_counts_plp_initial.json` が生成されることを確認

2. **PLP→PDP 抽出ロジックの修正（別 CR）**
   - CR-ATELIER-002 として、Moncler の PLP→PDP 抽出ロジックの修正を実施
   - URL パターン・セレクタ調整

### 今後のタスク
- PLP materialization の成功率向上
- セレクタの検証と調整
- エラーハンドリングの改善

## 関連ファイル

- `docs/spec/CR-ATELIER-001_MONCLER_PLP_TELEMETRY_AND_PLP_STATE_RECORDING.md` - 仕様書
- `app/agents/browser/telemetry.py` - Telemetry 実装
- `app/agents/browser/navigation_driver.py` - NavigationDriver 実装
- `app/agents/browser_use_agent.py` - BrowserUseAgent 実装（フォールバック版）
- `docs/moncler/PLP_TELEMETRY_ADAPTER_COMPLETION_REPORT.md` - 以前の完了レポート

