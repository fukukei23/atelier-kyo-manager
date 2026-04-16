# Stage 4: 残りのタスク一覧

**作成日時**: 2025-11-28  
**目的**: Phase 1完了後の残タスクを整理

---

## 1. 受け入れ条件の確認状況

### 1.1 構造面 ✅ **完了**

- [x] NavigationDriverが汎用的に処理（`_normalize_url()`, `_looks_like_trap_or_legal()`, `header_search_fallback()`, `_force_plp_recover()`）
- [x] BrowserUseAgentからMoncler固有のif文やハードコードがほぼ消えている
  - ✅ `/en-int/`, `forceLocale`, `moncler.com` URLのハードコードを削除
  - ✅ 重複メソッド（`_normalize_to_en_int_url()`等）を削除し、NavigationDriver経由に統一
- [x] 重複メソッドが削除されている
  - ✅ `_normalize_to_en_int_url()` → `_normalize_url()` ラッパーに置き換え
  - ✅ `_looks_like_trap_or_legal()` → NavigationDriver経由に統一
  - ✅ `_force_plp_recover()` → NavigationDriver経由に統一
  - ✅ `_plp_header_search_fallback()` → NavigationDriver経由に統一

### 1.2 設定面 ⚠️ **部分的に完了**

- [x] MONCLER_OFFICIALのsite_configが既存設定と互換性がある（コード修正完了）
- [ ] MONCLER_OFFICIALのsite_configが標準スキーマに沿っている（オプション）
- [ ] 他サイト（SSENSE等）も標準スキーマに沿っている（オプション）
- [ ] 新しいPLP Driverの標準スキーマだけで最低限リンク抽出が試せる（動作確認が必要）

### 1.3 観測性 ⚠️ **要確認**

- [ ] PLPナビゲーションの各段階でRunContextにログやスクリーンショットを残している
- [ ] 失敗時にSelectorDiscoveryAgentやSelfHealingAgentに渡せる情報が整理されている

---

## 2. 残っているタスク

### 2.1 必須タスク

#### ✅ Phase 1: NavigationDriverの汎用化（完了）
- [x] URL正規化ロジックの統一
- [x] Trap判定の汎用化
- [x] Header Search / PLP Recoveryの汎用化
- [x] BrowserUseAgentの重複ロジック削除
- [x] 既存設定との互換性確保

#### ⚠️ Phase 2: 動作確認と観測性の確認（未完了）

1. **MONCLER_OFFICIALの動作確認**
   - [ ] 実際にMONCLER_OFFICIALでPLP → PDP動作が正常に動作するか確認
   - [ ] 既存の動作が維持されているか確認
   - [ ] エラーログの確認

2. **観測性の確認**
   - [ ] PLPナビゲーションの各段階（URL生成、PLP安定化、PDP抽出）でログが出力されているか
   - [ ] スクリーンショットが適切に保存されているか
   - [ ] 失敗時に`failed_selectors`や`intent`が`run_context`に記録されているか

3. **他サイトでの動作確認（オプション）**
   - [ ] SSENSE等の他サイトで標準スキーマだけで動作するか確認
   - [ ] 不足している設定が明確になっているか

### 2.2 オプションタスク

#### Phase 3: 設定ファイルの標準化（オプション）

1. **MONCLER_OFFICIALの設定ファイル更新**
   - [ ] `overrides.local.json`のMONCLER_OFFICIAL設定を標準スキーマに合わせて更新
   - [ ] `locale.normalize_rules`を追加
   - [ ] `navigation.plp_recovery`を追加
   - [ ] `navigation.header_search.url_template`を追加

2. **他サイトの設定ファイル更新**
   - [ ] SSENSE, MATCHESFASHION等の設定を標準スキーマに合わせて更新

---

## 3. 次のステップ（優先順位順）

### 優先度: 高

1. **MONCLER_OFFICIALの動作確認**
   - 実際に動作させて、既存の動作が維持されているか確認
   - エラーが発生した場合は修正

2. **観測性の確認**
   - ログとスクリーンショットが適切に出力されているか確認
   - 失敗時の情報が`run_context`に記録されているか確認

### 優先度: 中

3. **他サイトでの動作確認**
   - SSENSE等で標準スキーマだけで動作するか確認
   - 不足している設定を明確にする

### 優先度: 低

4. **設定ファイルの標準化**
   - 既存設定との互換性があるため、必須ではない
   - 将来的な保守性向上のため推奨

---

## 4. 完了レポート作成

Phase 1完了時点での完了レポートを作成する必要があります：

- [ ] `docs/completion_reports/STAGE_4_PHASE_1_COMPLETION_REPORT.md` の作成
  - 実装日時
  - 概要
  - 実装ステップ
  - 変更ファイル一覧
  - 動作確認結果（動作確認後）
  - 設計上の改善点
  - 既知の制約・注意事項
  - 次のステップ

---

## 5. サマリー

**完了したタスク**:
- ✅ Phase 1: NavigationDriverの汎用化（コード実装）
- ✅ 既存設定との互換性確保

**残っているタスク**:
- ⚠️ Phase 2: 動作確認と観測性の確認
- ⚠️ 完了レポートの作成

**オプションタスク**:
- Phase 3: 設定ファイルの標準化（既存設定との互換性があるため必須ではない）

