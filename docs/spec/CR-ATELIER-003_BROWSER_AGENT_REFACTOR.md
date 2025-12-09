# CR-ATELIER-003 BrowserUseAgent リファクタリング

**Version:** 1.0

**Status:** Draft

**Owner:** Atelier Kyo / NexusCore Line

**Related CRs:** CR-ATELIER-002 (Step 1-8)

**Related Docs:**

- `docs/completion_reports/CR_ATELIER_002_STEP3_COMPLETION_REPORT.md`
- `docs/completion_reports/CR_ATELIER_002_STEP4_COMPLETION_REPORT.md`
- `docs/completion_reports/CR_ATELIER_002_STEP5_COMPLETION_REPORT.md`
- `docs/completion_reports/CR_ATELIER_002_STEP6_COMPLETION_REPORT.md`
- `docs/completion_reports/CR_ATELIER_002_STEP7_COMPLETION_REPORT.md`
- `docs/completion_reports/CR_ATELIER_002_STEP8_COMPLETION_REPORT.md`

---

## 1. Overview & Context

### 1.1 BrowserUseAgent が肥大化している問題

`app/agents/browser_use_agent.py` は現在 **2,761行** の巨大なファイルとなっており、以下の問題を抱えています：

1. **責務の混在**
   - PLP/ナビゲーション処理と PDP 抽出処理が混在
   - Moncler 固有の処理が複数箇所に散在
   - UI 操作ヘルパーとオーケストレーションロジックが混在

2. **NavigationDriver との重複**
   - `_ensure_plp_materialized` と `NavigationDriver.ensure_plp_materialized` が重複
   - `_collect_pdp_links` と `NavigationDriver.collect_pdp_links` が重複
   - `_plp_header_search_fallback` と `NavigationDriver.header_search_fallback` が重複
   - `_click_first_card_or_link` と `NavigationDriver.click_first_card_or_link` が重複
   - `_force_plp_recover` と `NavigationDriver.recover_plp` が重複

3. **Moncler 固有ロジックの散在**
   - `MONCLER_OFFICIAL` の分岐が複数箇所（42箇所）に存在
   - `moncler_plp_recovery` の呼び出しが複数箇所に存在
   - `MonclerDrissionHandler` の使用が1箇所に存在

4. **保守性の低下**
   - 変更の影響範囲が広く、テストが困難
   - コードレビューが困難
   - 新規サイト追加時の影響範囲が大きい

### 1.2 Stage 3〜7 で NavigationDriver / Self-Healing が導入済み

CR-ATELIER-002 Step 1〜8 までで、以下が実装済みです：

- **NavigationDriver**: PLP/ナビゲーション処理の集約
- **Self-Healing Agent**: 失敗分析と自動修復
- **Selector Discovery Agent**: セレクタ提案
- **Moncler Patch Builder**: パッチ候補生成
- **Telemetry**: 構造化された観測性

これらの機能は既に動作しており、BrowserUseAgent から NavigationDriver への移行は部分的に完了していますが、**レガシーコードが残存**しています。

### 1.3 Step 8 の目的

本リファクタリングの目的は、以下を達成することです：

1. **NavigationDriver への完全移行**
   - BrowserUseAgent 内の PLP/ナビゲーション処理を完全に NavigationDriver に移行
   - レガシーコードの削除

2. **Moncler 固有ロジックの専用モジュール化**
   - BrowserUseAgent から Moncler 固有の処理を排除
   - Moncler 専用のハンドラ/モジュールに集約

3. **オーケストレータとヘルパー群の物理分割**
   - BrowserUseAgent をオーケストレータと UI/ヘルパー群に分割
   - 責務境界を明確化

---

## 2. Scope (In-Scope / Out-of-Scope)

### 2.1 In-Scope（CR-ATELIER-003 でやること）

1. **NavigationDriver への完全移行**
   - `_ensure_plp_materialized` の削除（NavigationDriver に移行済み）
   - `_collect_pdp_links` の削除（NavigationDriver に移行済み）
   - `_plp_header_search_fallback` の削除（NavigationDriver に移行済み）
   - `_click_first_card_or_link` の削除（NavigationDriver に移行済み）
   - `_force_plp_recover` の削除（NavigationDriver に移行済み）
   - `_run_plp_flow` 内の旧ロジックの削除（NavigationDriver 経由のみに統一）

2. **Moncler 固有ロジックの専用モジュール化**
   - `MONCLER_OFFICIAL` 分岐の削除
   - `moncler_plp_recovery` 呼び出しの整理
   - `MonclerDrissionHandler` の使用箇所の整理
   - Moncler 専用ハンドラの導入/強化

3. **オーケストレータとヘルパー群の物理分割**
   - `browser_orchestrator.py` の新規作成（高レベルフロー制御）
   - `ui_helpers.py` の拡張（低レベル UI 操作）
   - `browser_use_agent.py` を薄い Facade として残す

### 2.2 Out-of-Scope（CR-ATELIER-003 ではやらないこと）

- **Flask UI (app/templates/**, app/routes.py 等)**
  - `.cursorrules` の編集ポリシーに従い、編集禁止

- **新規サイト追加**
  - Moncler 以外のサイトへの一般化は別 CR の対象

- **LLM 連携の設計変更**
  - LLM 連携部分は変更しない

- **Telemetry スキーマの変更**
  - 既存の Telemetry スキーマは変更しない（互換性維持）

---

## 3. Implementation Plan

### Phase A: NavigationDriver への移行完了＋レガシー削除

#### Phase A-1: 呼び出し経路の統一

**目的**: BrowserUseAgent 内で PLP/ナビゲーションを行う箇所を、すべて NavigationDriver 経由に統一する。

**対象メソッド**:
- `_ensure_plp_materialized`: NavigationDriver.ensure_plp_materialized に移行済み → 削除
- `_collect_pdp_links`: NavigationDriver.collect_pdp_links に移行済み → 削除
- `_plp_header_search_fallback`: NavigationDriver.header_search_fallback に移行済み → 削除
- `_click_first_card_or_link`: NavigationDriver.click_first_card_or_link に移行済み → 削除
- `_force_plp_recover`: NavigationDriver.recover_plp に移行済み → 削除

**実装手順**:
1. `git grep` で各メソッドの参照箇所を確認
2. すべての呼び出しを NavigationDriver 経由に置き換え
3. メソッド定義を削除
4. テストを実行して動作確認

#### Phase A-2: `_run_plp_flow` 内の旧ロジック削除

**目的**: `_run_plp_flow` 内の旧ロジックを削除し、NavigationDriver 経由のみに統一する。

**対象箇所**:
- `_run_plp_flow` 内の `_ensure_plp_materialized` 呼び出し
- `_run_plp_flow` 内の `_collect_pdp_links` 呼び出し
- `_run_plp_flow` 内の `_plp_header_search_fallback` 呼び出し
- `_run_plp_flow` 内の `_click_first_card_or_link` 呼び出し
- `_run_plp_flow` 内の `_force_plp_recover` 呼び出し
- `_run_plp_flow` 内の trap 判定ロジック（NavigationDriver が処理済み）

**実装手順**:
1. `_run_plp_flow` 内の NavigationDriver 呼び出しを確認
2. 旧ロジックの条件分岐を削除
3. NavigationDriver の結果をそのまま使用するように変更
4. テストを実行して動作確認

#### Phase A-3: テスト確認

**実行コマンド**:
```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate
python -m pytest tests/test_plp_driver.py tests/test_browser_use_agent_plp_integration.py -q -v
```

**確認事項**:
- 既存テストの期待値（ログメッセージ、挙動）が壊れていないか
- NavigationDriver 経由の呼び出しが正しく動作しているか

### Phase B: Moncler 固有ロジックの専用モジュール化

#### Phase B-1: サイト別ハンドラの導入/強化

**目的**: Moncler 専用の処理を `MonclerHandler`（仮）クラスに集約する。

**対象モジュール**:
- `app/agents/browser_use_moncler_patch.py`: 既存の Moncler パッチ
- `app/agents/moncler_patch_builder.py`: パッチ生成ユーティリティ
- `app/agents/plugins/moncler_plp_v1.py`: Moncler PLP 戦略

**実装手順**:
1. `MonclerHandler` クラスを新規作成（または既存の `browser_use_moncler_patch.py` を拡張）
2. Moncler 専用処理を `MonclerHandler` に集約
3. BrowserUseAgent からは、サイトキー → ハンドラの解決 → ハンドラ呼び出しのみを行う

#### Phase B-2: Moncler 分岐の削除

**目的**: BrowserUseAgent 内の Moncler 専用 if 分岐を削除し、ハンドラ呼び出しに置き換える。

**対象箇所**:
- `MONCLER_OFFICIAL` の分岐（42箇所）
- `moncler_plp_recovery` の呼び出し（複数箇所）
- `MonclerDrissionHandler` の使用（1箇所）

**実装手順**:
1. `MONCLER_OFFICIAL` 分岐を特定
2. ハンドラ呼び出しに置き換え
3. 元コードを削除
4. テストを実行して動作確認

#### Phase B-3: テスト確認

**実行コマンド**:
```bash
python -m pytest \
  tests/test_moncler_pdp_url.py \
  tests/test_moncler_self_healing.py \
  tests/test_moncler_selector_discovery.py \
  tests/test_moncler_patch_builder.py \
  -q -v
```

**確認事項**:
- Moncler 関連テストがすべてパスすること
- Moncler 固有ロジックが正しく動作しているか

### Phase C: オーケストレータとヘルパー群の物理分割

#### Phase C-1: `browser_orchestrator.py` の新規作成

**目的**: 高レベルフロー制御を `BrowserRunOrchestrator` クラスに分離する。

**対象メソッド**:
- `run`: メイン実行フロー
- `_run_plp_flow`: PLP フロー実行
- `_run_pdp_flow`: PDP フロー実行
- `_run_learning_flow`: 学習フロー実行
- `_handle_run_failure`: 失敗処理

**実装手順**:
1. `app/agents/browser/browser_orchestrator.py` を新規作成
2. `BrowserRunOrchestrator` クラスを定義
3. 高レベルフロー制御メソッドを移動
4. BrowserUseAgent から `BrowserRunOrchestrator` を呼び出すように変更

#### Phase C-2: `ui_helpers.py` の拡張

**目的**: 低レベル UI 操作を `ui_helpers.py` に集約する。

**対象メソッド**:
- `_dismiss_geo_modal`: 地理モーダルの閉じる
- `_accept_cookies_if_present`: Cookie 受け入れ
- `_kill_overlays`: オーバーレイの削除
- `_click_continue_shopping_if_present`: Continue Shopping ボタンのクリック
- `safe_wait_selector`: セレクタの安全な待機
- その他の `safe_*` 系メソッド

**実装手順**:
1. `ui_helpers.py` を確認（既に存在）
2. BrowserUseAgent 内の UI 操作メソッドを `ui_helpers.py` に移動
3. BrowserUseAgent から `ui_helpers` を import して使用

#### Phase C-3: `browser_use_agent.py` を薄い Facade として残す

**目的**: BrowserUseAgent を Public API を維持する薄い Facade として残す。

**実装手順**:
1. `BrowserUseAgent` クラスを薄い Facade に変更
2. `run()` メソッドは `BrowserRunOrchestrator` を呼び出すだけにする
3. Public API（`run()` など）は互換性を保つ

#### Phase C-4: 依存関係の整理

**目的**: 循環参照が起きないように、モジュール間の依存関係を一方向に揃える。

**依存関係**:
```
BrowserUseAgent → BrowserRunOrchestrator → SessionManager, NavigationDriver, TelemetryService, UiHelpers
```

**実装手順**:
1. 各モジュールの import を確認
2. 循環参照が発生しないように調整
3. 必要に応じて、インターフェースを導入

#### Phase C-5: テスト確認

**実行コマンド**:
```bash
python -m pytest -q -v
```

**確認事項**:
- 物理分割後も、既存テストがパスすること
- 必要であれば、`BrowserRunOrchestrator` 単体のテストを追加

---

## 4. Testing Strategy

### 4.1 既存テストの維持

**方針**: すべての既存テストがグリーンであることを確認する。

**対象テスト**:
- `tests/test_plp_driver.py`
- `tests/test_browser_use_agent_plp_integration.py`
- `tests/test_moncler_pdp_url.py`
- `tests/test_moncler_self_healing.py`
- `tests/test_moncler_selector_discovery.py`
- `tests/test_moncler_patch_builder.py`
- その他の既存テスト

**実行コマンド**:
```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate
python -m pytest -q -v
```

### 4.2 必要に応じて追加するテスト

**NavigationDriver 単体テスト**:
- `NavigationDriver.run_plp_flow` の単体テスト
- `NavigationDriver.collect_pdp_links` の単体テスト

**BrowserUseAgent の薄いラッパーとしての統合テスト**:
- `BrowserUseAgent.run()` が `BrowserRunOrchestrator` を正しく呼び出すことを確認
- Public API の互換性を確認

**MonclerHandler のテスト**:
- `MonclerHandler` が Moncler 専用処理を正しく実行することを確認

---

## 5. Risks & Mitigation

### 5.1 既存 run の挙動が変わるリスク

**リスク**: NavigationDriver への移行により、既存の run の挙動が変わる可能性がある。

**緩和策**:
- 段階的に移行し、各 Phase でテストを実行
- 既存のログメッセージを維持（可能な限り）
- 変更前後で挙動が変わる可能性がある箇所は、コメントかレポートで必ず説明

### 5.2 Moncler 固有ロジックの移行ミス

**リスク**: Moncler 固有ロジックの移行時に、処理が漏れる可能性がある。

**緩和策**:
- `git grep` で `MONCLER_OFFICIAL` 分岐をすべて特定
- Moncler 関連テストをすべて実行して確認
- 移行前後で Moncler run を実行して動作確認

### 5.3 ログ・Telemetry が変わりすぎることによるデバッグ困難

**リスク**: ログメッセージや Telemetry キーを変更すると、既存のレポート類との互換性が失われる。

**緩和策**:
- ログメッセージや Telemetry キーを変更する場合は、既存のレポート類との互換性に配慮
- 変更が必要な場合は、バージョン管理を導入

### 5.4 循環参照の発生

**リスク**: モジュール分割時に循環参照が発生する可能性がある。

**緩和策**:
- 依存関係を一方向に揃える
- 必要に応じて、インターフェースを導入
- `TYPE_CHECKING` を使用して循環参照を回避

---

## 6. Acceptance Criteria

CR-ATELIER-003 が完了したと見なす条件は以下：

1. **NavigationDriver への完全移行**
   - BrowserUseAgent 内の PLP/ナビゲーション処理が完全に NavigationDriver に移行
   - レガシーコード（`_ensure_plp_materialized`, `_collect_pdp_links`, `_plp_header_search_fallback`, `_click_first_card_or_link`, `_force_plp_recover`）が削除されている

2. **Moncler 固有ロジックの専用モジュール化**
   - BrowserUseAgent から Moncler 固有の処理が排除されている
   - Moncler 専用のハンドラ/モジュールに集約されている

3. **オーケストレータとヘルパー群の物理分割**
   - `browser_orchestrator.py` が新規作成されている
   - `ui_helpers.py` が拡張されている
   - `browser_use_agent.py` が薄い Facade として残っている

4. **既存テストがすべてパス**
   - すべての既存テストがグリーンであること

5. **`.cursorrules` / README のポリシーを破る変更がない**
   - UI テンプレート編集等の禁止事項に違反していないこと

---

## 7. 今後の拡張（CR-ATELIER-004 以降候補）

- Moncler 以外のサイトへの一般化（サイト別ハンドラの共通化）
- さらに細かい UI ヘルパーの切り出し
- BrowserRunOrchestrator の単体テスト追加
- パフォーマンス最適化

