# PLP Telemetry Adapter 完了レポート

## 実装日時
2025-12-04

## 概要

Moncler PLP 実行時に `plp_dom_initial_materialized.html` と `selector_counts_plp_initial.json` が生成されない問題を解決するため、`TelemetryClient` / `TelemetryService` に共通メソッド `record_plp_state` を追加し、`NavigationDriver` からの呼び出しを統一しました。

### 目的
- PLP 初期状態の DOM スナップショットとセレクタカウントを確実に記録する
- Telemetry インターフェースの統一により、呼び出し側の複雑な分岐を排除
- Phase 1 のゴール「壊れ方を観測できる状態」を実現

### 原則
- 既存の Telemetry API (`save_dom`, `save_json`, `save_screenshot`, `write_fail_snapshot`) との互換性を維持
- 例外は内部で握りつぶさず、logger.warn でログを残す
- ファイル名は固定（`plp_dom_initial_materialized.html`, `selector_counts_plp_initial.json`）

## 実装ステップ

### Step 1: TelemetryClient に `record_plp_state` メソッドを追加

**ファイル**: `app/agents/browser/telemetry.py`

**変更内容**:
- `TelemetryClient` クラスに `record_plp_state` メソッドを追加
- 内部で `TelemetryService.record_plp_state` を呼び出すラッパーとして実装
- 例外発生時は logger.warn でログを残し、呼び出し元に例外を伝播させない

**コード例**:
```python
async def record_plp_state(
    self,
    page: Any,
    *,
    name: str = "plp_dom_initial_materialized",
    selectors: Optional[List[str]] = None,
    site_config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    PLP 初期状態の DOM とセレクタカウントを保存するための共通 API（Phase1 Moncler 診断用）
    
    Args:
        page: Playwright Page オブジェクト
        name: 保存ファイル名のベース（デフォルト: "plp_dom_initial_materialized"）
        selectors: セレクタカウント対象（オプション）
        site_config: サイト設定（selectors が None の場合、ここから自動取得）
    
    保存されるファイル:
    - plp_dom_initial_materialized.html（DOM スナップショット）
    - selector_counts_plp_initial.json（セレクタカウント、selectors が指定された場合のみ）
    """
    try:
        await self._service.record_plp_state(
            page=page,
            name=name,
            selectors=selectors,
            site_config=site_config,
        )
    except Exception as e:
        self.logger.warning(f"[TelemetryClient] Failed to record PLP state '{name}': {e}", exc_info=True)
```

### Step 2: TelemetryService の `record_plp_state` のファイル名生成ロジックを修正

**ファイル**: `app/agents/browser/telemetry.py`

**変更内容**:
- `name` が `"plp_dom_initial"` または `"plp_dom_initial_materialized"` の場合、ファイル名を固定
- DOM: `plp_dom_initial_materialized.html`
- セレクタカウント: `selector_counts_plp_initial.json`

**変更前**:
```python
await self._save_dom(page, name)
if selectors:
    await self._count_selectors(page, selectors, name=f"selector_counts_{name}")
```

**変更後**:
```python
# DOM保存（ファイル名を固定: plp_dom_initial_materialized.html）
if name == "plp_dom_initial" or name == "plp_dom_initial_materialized":
    await self._save_dom(page, "plp_dom_initial_materialized")
else:
    await self._save_dom(page, name)

# セレクタカウント（指定がある場合、ファイル名を固定: selector_counts_plp_initial.json）
if selectors:
    if name == "plp_dom_initial" or name == "plp_dom_initial_materialized":
        await self._count_selectors(page, selectors, name="selector_counts_plp_initial")
    else:
        await self._count_selectors(page, selectors, name=f"selector_counts_{name}")
```

### Step 3: NavigationDriver の呼び出しロジックを簡略化

**ファイル**: `app/agents/browser/navigation_driver.py`

**変更内容**:
- `hasattr(telemetry, "_service")` や `hasattr(telemetry, "record_plp_state")` などの複雑な分岐を削除
- Telemetry が `None` でない場合は、直接 `await self.telemetry.record_plp_state(...)` を呼ぶ実装に統一
- 「Telemetry object does not have expected interface」系の警告ログを削除

**変更前** (約50行の複雑な分岐):
```python
if condition_result:
    logger.info(f"[NavigationDriver] Inside if block: condition_result={condition_result}")
    try:
        if not self.telemetry:
            logger.warning("[NavigationDriver] Telemetry not available, skipping PLP state recording")
        else:
            telemetry_type = type(self.telemetry).__name__
            logger.debug(f"[NavigationDriver] Telemetry type: {telemetry_type}")
            
            telemetry_service = None
            
            if hasattr(self.telemetry, '_service'):
                telemetry_service = self.telemetry._service
                logger.debug("[NavigationDriver] Using TelemetryClient._service")
            elif hasattr(self.telemetry, 'record_plp_state'):
                telemetry_service = self.telemetry
                logger.debug(f"[NavigationDriver] Using TelemetryService directly (type: {telemetry_type})")
            else:
                logger.warning(f"[NavigationDriver] Telemetry object ({telemetry_type}) does not have expected interface")
                logger.warning(f"[NavigationDriver] Available attributes: {dir(self.telemetry)}")
            
            if telemetry_service:
                try:
                    # ... 複雑な処理 ...
                except Exception as telem_e:
                    logger.warning(f"[NavigationDriver] Failed to record PLP state: {telem_e}", exc_info=True)
            else:
                logger.warning("[NavigationDriver] Telemetry service not available, skipping PLP state recording")
    except Exception as outer_telem_e:
        logger.error(f"[NavigationDriver] Outer exception during PLP state recording attempt: {outer_telem_e}", exc_info=True)
```

**変更後** (約15行のシンプルな実装):
```python
condition_result = materialized or tiles_detected
if condition_result and self.telemetry:
    try:
        # site_config からセレクタを取得
        pdp_cfg = (ctx.site_config.get("selectors") or {}).get("pdp", {}) or {}
        selectors = (
            (pdp_cfg.get("pdp_link_selectors") or []) +
            (pdp_cfg.get("plp_container_selectors") or [])
        )
        logger.info(f"[NavigationDriver] Recording PLP state: materialized={materialized}, tiles_detected={tiles_detected}")
        await self.telemetry.record_plp_state(
            self.page,
            name="plp_dom_initial_materialized",
            selectors=selectors if selectors else None,
            site_config=ctx.site_config,
        )
        logger.info("[NavigationDriver] Saved PLP DOM snapshot and selector counts")
    except Exception as e:
        logger.warning(f"[NavigationDriver] Failed to record PLP state: {e}", exc_info=True)
elif not condition_result:
    logger.debug(f"[NavigationDriver] Skipping PLP state recording: materialized={materialized}, tiles_detected={tiles_detected}")
elif not self.telemetry:
    logger.warning("[NavigationDriver] Telemetry not available, skipping PLP state recording")
```

## 変更ファイル一覧

### 新規作成ファイル
- `.spec/CR-AKM-001_MONCLER_PLP_TELEMETRY_AND_PLP_STATE_RECORDING.md` - 仕様書

### 変更ファイル
- `app/agents/browser/telemetry.py`
  - `TelemetryClient.record_plp_state` メソッドを追加
  - `TelemetryService.record_plp_state` のファイル名生成ロジックを修正
- `app/agents/browser/navigation_driver.py`
  - PLP state recording の呼び出しロジックを簡略化（約50行 → 約15行）

## 動作確認結果

### 静的解析結果
- リンターエラー: なし
- 型チェッカー: 未実行（必要に応じて後で実行）

### コードレビュー結果
- ✅ `TelemetryClient` と `TelemetryService` の両方に `record_plp_state` が実装され、同じインタフェースを提供
- ✅ 既存の Telemetry API (`save_dom`, `save_json`, `save_screenshot`, `write_fail_snapshot`) との互換性を維持
- ✅ 例外処理が適切に実装され、呼び出し元に例外を伝播させない
- ✅ `NavigationDriver` の呼び出しロジックが簡略化され、可読性が向上

### テスト結果
**実行待ち**: 以下のコマンドで動作確認が必要です。

```bash
python -m app.scripts.run_site moncler --query "down jacket" --headful
```

**確認項目**:
1. 最新の `run_id` ディレクトリ配下に以下が生成されること:
   - `plp_dom_initial_materialized.html`
   - `selector_counts_plp_initial.json`
2. `system.log` に以下の警告が出ていないこと:
   - `Telemetry object (TelemetryClient/TelemetryService) does not have expected interface`
   - `Telemetry service not available, skipping PLP state recording`
3. `tiles_detected=True` となった run で、PLP スナップショットファイルが必ず生成されること

## 設計上の改善点

### アーキテクチャの改善
- **インタフェースの統一**: `TelemetryClient` と `TelemetryService` の両方に `record_plp_state` を実装することで、呼び出し側が型を意識せずに同じメソッドを呼べるようになった
- **責務の明確化**: `NavigationDriver` から Telemetry の内部実装詳細（`_service` 属性など）を隠蔽し、インタフェース経由でのみアクセスする設計に統一

### 将来の拡張性への配慮
- `record_plp_state` メソッドは `name` パラメータを受け取るため、将来的に異なる名前の PLP スナップショットを保存することも可能
- `selectors` が `None` の場合は JSON を保存しないため、DOM のみを保存する用途にも対応可能

### コード品質の向上
- `NavigationDriver` の呼び出しロジックが約50行から約15行に簡略化され、可読性と保守性が向上
- 複雑な `hasattr` チェックを削除し、シンプルな条件分岐に統一

## 既知の制約・注意事項

### 既存コードとの互換性
- ✅ 既存の Telemetry API (`save_dom`, `save_json`, `save_screenshot`, `write_fail_snapshot`) は変更していないため、既存コードへの影響はない
- ✅ `NavigationDriver` の変更は、PLP state recording の呼び出し部分のみであり、他の機能への影響はない

### 制限事項やトレードオフ
- ファイル名は `name` パラメータが `"plp_dom_initial"` または `"plp_dom_initial_materialized"` の場合のみ固定される
- 他の `name` 値の場合は、従来通り `name` ベースのファイル名が生成される

### 移行時の注意点
- 既存の run ディレクトリには影響しない（新規実行時のみ適用）
- `TelemetryService` が直接渡される場合も、`record_plp_state` メソッドが存在するため動作する

## 次のステップ

### 即座に実施すべきこと
1. **動作確認テストの実行**
   ```bash
   python -m app.scripts.run_site moncler --query "down jacket" --headful
   ```
   - 最新の `run_id` ディレクトリを確認
   - `plp_dom_initial_materialized.html` と `selector_counts_plp_initial.json` が生成されていることを確認
   - `system.log` に警告が出ていないことを確認

### 今後のタスク（CR-AKM-002予定）
- Moncler の PLP→PDP 抽出ロジックの修正（URL パターン・セレクタ調整）
- セレクタの検証と調整
- PLP マテリアライゼーションの改善

## 関連ファイル

- `.spec/CR-AKM-001_MONCLER_PLP_TELEMETRY_AND_PLP_STATE_RECORDING.md` - 仕様書
- `app/agents/browser/telemetry.py` - Telemetry 実装
- `app/agents/browser/navigation_driver.py` - NavigationDriver 実装
- `docs/moncler/PHASE1_STATUS_SUMMARY.md` - Phase 1 ステータスサマリー
- `docs/moncler/PLP_EXTRACTION_FIX_TASK_TEMPLATE.md` - PLP 抽出修正タスクテンプレート

