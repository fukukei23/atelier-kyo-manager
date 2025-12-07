# CR-ATELIER-001 Hotfix: TelemetryClient.record_plp_state 実装の確実な反映

- **Status:** In-Progress
- **Author:** [AI Assistant]
- **Date:** 2025-12-04
- **Related CR:** CR-ATELIER-001_MONCLER_PLP_TELEMETRY_AND_PLP_STATE_RECORDING.md

## 1. Overview & Context

### 目的 (Why)
Moncler PLP 実行時に `plp_dom_initial_materialized.html` と `selector_counts_plp_initial.json` が生成されない問題を解決するため、`TelemetryClient.record_plp_state` の実装を確実に反映し、動作を検証する。

### 背景 (Background)
- 最新 run (`instance/runs/20251204_234612_007`) では、`*plp*` ファイルが 0 件
- `system.log` も存在しない → run が初期化前に落ちている可能性
- 以前のログでは `'TelemetryClient' object has no attribute 'record_plp_state'` が発生していた
- `TelemetryClient.record_plp_state` は実装されているが、実際にファイルが生成されていない

### 現状の問題
1. **実装は存在するが動作していない**
   - `app/agents/browser/telemetry.py` の `TelemetryClient.record_plp_state` は実装済み（487-516行目）
   - `TelemetryService.record_plp_state` も実装済み（89-138行目）
   - しかし、実際の run ではファイルが生成されていない

2. **run が初期化前に落ちている可能性**
   - `system.log` が存在しない → `RunContext` の初期化前にエラーが発生している可能性

3. **実装の検証が不十分**
   - ログでは `record_plp_state` が呼ばれているが、ファイル生成まで確認できていない

### 参照 (References)
- `docs/spec/CR-ATELIER-001_MONCLER_PLP_TELEMETRY_AND_PLP_STATE_RECORDING.md` - 元の仕様書
- `app/agents/browser/telemetry.py` - Telemetry 実装
- `app/agents/browser/navigation_driver.py` - NavigationDriver 実装
- `app/agents/browser_use_agent.py` - BrowserUseAgent 実装

## 2. Scope

### ✅ In-Scope (やること)

1. **TelemetryClient.record_plp_state の実装確認と修正**
   - 現在の実装を確認し、`NavigationDriver` からの呼び出しシグネチャと完全に一致することを確認
   - ファイル生成ロジックが正しく動作することを確認
   - エラーハンドリングが適切であることを確認

2. **TelemetryService.record_plp_state の実装確認と修正**
   - DOM 保存ロジック（`_save_dom`）が正しく動作することを確認
   - セレクタカウント保存ロジック（`_count_selectors`）が正しく動作することを確認
   - ファイル名が固定値（`plp_dom_initial_materialized.html`, `selector_counts_plp_initial.json`）になることを確認

3. **動作確認と検証**
   - Moncler run を実行し、ファイルが生成されることを確認
   - `system.log` にエラーが出ていないことを確認
   - `AttributeError` が発生していないことを確認

### ❌ Out-of-Scope (やらないこと)

- PLP→PDP 抽出ロジックの修正（別 CR）
- 他サイトの Telemetry 拡張
- PLP materialization ロジックの再設計

## 3. Implementation Plan

### Step 1: 現在の実装状況の確認

1. **TelemetryClient.record_plp_state の実装確認**
   - `app/agents/browser/telemetry.py` の 487-516行目を確認
   - シグネチャが `NavigationDriver` からの呼び出しと一致しているか確認
   - `TelemetryService.record_plp_state` への委譲が正しく行われているか確認

2. **TelemetryService.record_plp_state の実装確認**
   - `app/agents/browser/telemetry.py` の 89-138行目を確認
   - `_save_dom` と `_count_selectors` の呼び出しが正しいか確認
   - ファイル名が固定値になっているか確認

3. **NavigationDriver からの呼び出し確認**
   - `app/agents/browser/navigation_driver.py` の 251-256行目を確認
   - 呼び出しシグネチャが正しいか確認

### Step 2: 実装の修正（必要に応じて）

#### 2.1 TelemetryClient.record_plp_state の修正

**期待されるシグネチャ:**
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

**実装要件:**
- `TelemetryService.record_plp_state` に委譲する
- 例外発生時は `logger.warning` でログを残し、呼び出し元に例外を伝播させない
- `exc_info=True` を指定してスタックトレースを記録

#### 2.2 TelemetryService.record_plp_state の修正

**期待されるシグネチャ:**
```python
async def record_plp_state(
    self,
    page: "Page",
    *,
    name: str = "plp_dom_initial",
    selectors: Optional[List[str]] = None,
    site_config: Optional[Dict[str, Any]] = None,
) -> None:
```

**実装要件:**
- `name` が `"plp_dom_initial"` または `"plp_dom_initial_materialized"` の場合、ファイル名を固定:
  - DOM: `plp_dom_initial_materialized.html`
  - セレクタカウント: `selector_counts_plp_initial.json`
- `_save_dom` を呼び出して DOM を保存
- `selectors` が指定された場合、`_count_selectors` を呼び出してセレクタカウントを保存
- 例外発生時は `logger.warning` でログを残す

#### 2.3 _save_dom と _count_selectors の確認

**期待される動作:**
- `_save_dom(page, name)` → `run_context.save_content(f"{name}.html", html)` を呼び出す
- `_count_selectors(page, selectors, name=name)` → `run_context.save_json(f"{name}.json", counts)` を呼び出す

### Step 3: 動作確認

1. **Moncler run の実行**
   ```bash
   cd /home/yn441611/atelier-kyo-manager
   source venv/bin/activate
   python -m app.scripts.run_site moncler --query "down jacket" --headful
   ```

2. **最新 run ディレクトリの確認**
   ```bash
   LATEST=$(ls -td instance/runs/2025* | head -1)
   echo "LATEST RUN = $LATEST"
   ls -la "$LATEST"/*plp* 2>/dev/null
   ls -la "$LATEST/system.log" 2>/dev/null
   ```

3. **期待される結果**
   - ✅ `plp_dom_initial_materialized.html` が存在する
   - ✅ `selector_counts_plp_initial.json` が存在する
   - ✅ `system.log` が存在する
   - ✅ `system.log` に `AttributeError` が出ていない
   - ✅ `system.log` に `[NavigationDriver] Saved PLP DOM snapshot and selector counts` が記録されている

### Step 4: 問題が発生した場合の切り分け

#### 4.1 system.log が存在しない場合
- `RunContext` の初期化前にエラーが発生している可能性
- `python -m app.scripts.run_site ...` のコンソールログ全体を確認
- `ls -la "$LATEST"` で run ディレクトリ直下のファイル一覧を確認
- どのフェーズで落ちているかを切り分ける

#### 4.2 AttributeError が発生している場合
- `TelemetryClient.record_plp_state` の実装が正しく反映されていない可能性
- Python のキャッシュ（`.pyc` ファイル）をクリア
- 仮想環境が正しく有効化されているか確認

#### 4.3 ファイルが生成されない場合
- `TelemetryService.record_plp_state` の実装を確認
- `_save_dom` と `_count_selectors` の実装を確認
- `run_context.save_content` と `run_context.save_json` が正しく動作しているか確認

## 4. Testing Strategy

### テスト方針
- E2E に近い「実行テスト」を中心にする
- Moncler 向け `run_site` を実行し、実際に run ディレクトリ配下のファイル生成を確認

### テストコマンド
```bash
# 1. 仮想環境の有効化
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate

# 2. Moncler run の実行
python -m app.scripts.run_site moncler --query "down jacket" --headful

# 3. 最新 run ディレクトリの確認
LATEST=$(ls -td instance/runs/2025* | head -1)
echo "LATEST RUN = $LATEST"

# 4. ファイル生成の確認
ls -la "$LATEST"/*plp* 2>/dev/null
ls -la "$LATEST/system.log" 2>/dev/null

# 5. ログの確認
grep -E "record_plp_state|Recording PLP state|Saved PLP DOM|Failed to record PLP state|AttributeError" "$LATEST/system.log" 2>/dev/null | tail -20
```

### 主な検証観点

#### 正常系
- ✅ `tiles_detected=True` となった run で、PLP スナップショットファイルが必ず生成される
- ✅ `plp_dom_initial_materialized.html` が存在し、サイズが 0 より大きい
- ✅ `selector_counts_plp_initial.json` が存在し、有効な JSON 形式である
- ✅ `system.log` に `[NavigationDriver] Saved PLP DOM snapshot and selector counts` が記録されている

#### 異常系
- ✅ Telemetry が `None` の場合、NavigationDriver はエラーにならずに処理を継続する
- ✅ Telemetry 内部で例外が出ても、run 全体がクラッシュしない（ログに WARN が残るだけ）
- ✅ `AttributeError: 'TelemetryClient' object has no attribute 'record_plp_state'` が発生しない

#### 既存機能への影響
- ✅ `save_dom` / `save_json` / `write_fail_snapshot` を利用している既存コードに影響がないこと

## 5. 実装チェックリスト

### TelemetryClient.record_plp_state
- [ ] シグネチャが `NavigationDriver` からの呼び出しと完全に一致している
- [ ] `TelemetryService.record_plp_state` に正しく委譲している
- [ ] 例外発生時は `logger.warning` でログを残している
- [ ] `exc_info=True` を指定してスタックトレースを記録している

### TelemetryService.record_plp_state
- [ ] `name` が `"plp_dom_initial"` または `"plp_dom_initial_materialized"` の場合、ファイル名を固定している
- [ ] `_save_dom` を正しく呼び出している
- [ ] `selectors` が指定された場合、`_count_selectors` を正しく呼び出している
- [ ] 例外発生時は `logger.warning` でログを残している

### _save_dom と _count_selectors
- [ ] `_save_dom` が `run_context.save_content` を正しく呼び出している
- [ ] `_count_selectors` が `run_context.save_json` を正しく呼び出している
- [ ] ファイル名が正しく生成されている

### NavigationDriver
- [ ] `self.telemetry.record_plp_state` の呼び出しシグネチャが正しい
- [ ] 例外発生時は `logger.warning` でログを残している

## 6. 次のステップ

1. **実装の確認と修正**
   - 上記チェックリストに従って実装を確認
   - 必要に応じて修正を実施

2. **動作確認の実行**
   - Moncler run を実行し、ファイル生成を確認

3. **問題が発生した場合の切り分け**
   - `system.log` が存在しない場合 → RunContext 初期化前の問題を調査
   - `AttributeError` が発生している場合 → 実装の反映を確認
   - ファイルが生成されない場合 → TelemetryService の実装を確認

4. **完了レポートの作成**
   - 実装完了後、`docs/spec/CR-ATELIER-001_HOTFIX_TELEMETRY_RECORD_PLP_STATE_IMPLEMENTATION_COMPLETION_REPORT.md` を作成

