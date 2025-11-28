# Stage 3A-2 お片付けレポート

## 実施日
2025年1月（Stage 3A-2 完了後）

## 確認項目と結果

### 1. BrowserUseAgent の private メソッド確認

#### ✅ `_collect_pdp_links`
- **状態**: 薄いラッパーとして正しく実装されている
- **実体**: `NavigationDriver.collect_pdp_links` に移行済み
- **使用状況**: `_run_plp_flow` 内で使用されているが、`NavigationDriver.run_plp_flow` の結果（`nav_outcome.pdp_links`）を優先的に使用
- **TODO**: Stage 3A-2 完了後、すべての呼び出しが NavigationDriver 経由になったら削除可能

#### ⚠️ `_ensure_plp_materialized`
- **状態**: 実体は `NavigationDriver.ensure_plp_materialized` に移行済み
- **使用状況**: NavigationDriver が使われていない場合のフォールバックとして残っている
- **使用箇所**: 
  - `_run_plp_flow` 内で NavigationDriver が materialize を実行していない場合
  - ヘッダ検索 fallback 後の materialize
- **TODO**: Stage 3A-2 完了後、NavigationDriver が常に使われるようになったら削除可能

#### ⚠️ `_force_plp_recover`
- **状態**: 実体は `NavigationDriver.recover_plp` に移行済み
- **使用状況**: NavigationDriver が使われていない場合のフォールバックとして残っている
- **使用箇所**: 
  - `_run_plp_flow` 内で NavigationDriver が trap 判定・復旧を実行していない場合
  - `_ensure_plp_materialized` 内でのロケール回復
- **TODO**: Stage 3A-2 完了後、NavigationDriver が常に使われるようになったら削除可能

#### ⚠️ `_plp_header_search_fallback`
- **状態**: 実体は `NavigationDriver.header_search_fallback` に移行済み
- **使用状況**: NavigationDriver が使われていない場合のフォールバックとして残っている
- **使用箇所**: `_run_plp_flow` 内で NavigationDriver が fallback を実行していない場合
- **TODO**: Stage 3A-2 完了後、NavigationDriver が常に使われるようになったら削除可能

#### ⚠️ `_click_first_card_or_link`
- **状態**: 実体は `NavigationDriver.click_first_card_or_link` に移行済み
- **使用状況**: NavigationDriver が使われていない場合のフォールバックとして残っている
- **使用箇所**: `_run_plp_flow` 内で NavigationDriver が fallback を実行していない場合
- **TODO**: Stage 3A-2 完了後、NavigationDriver が常に使われるようになったら削除可能

### 2. NavigationOutcome フィールド確認

すべてのフィールドが使用されています：

- ✅ `entry_url`: NavigationDriver と BrowserUseAgent で使用
- ✅ `plp_materialized`: BrowserUseAgent で materialize スキップ判定に使用
- ✅ `trap_detected`: BrowserUseAgent で trap 判定スキップに使用
- ✅ `trap_reason`: BrowserUseAgent でログ出力に使用
- ✅ `recovered`: BrowserUseAgent で復旧状態確認に使用
- ✅ `pdp_links`: BrowserUseAgent で PDP リンク取得に使用
- ✅ `fallback_used`: BrowserUseAgent で fallback 重複実行スキップに使用

**結論**: 未使用フィールドなし

### 3. NavigationContext フィールド確認

すべてのフィールドが使用されています：

- ✅ `site`: NavigationDriver で使用
- ✅ `query`: NavigationDriver で使用（header_search_fallback など）
- ✅ `site_config`: NavigationDriver で使用
- ✅ `settings`: NavigationDriver で使用
- ✅ `run_context`: NavigationDriver で使用（screenshot など）
- ✅ `start_t`: NavigationDriver で使用（時間管理）
- ✅ `budget_ms`: NavigationDriver で使用（時間管理）
- ✅ `entry_url`: NavigationDriver で使用（初期 URL）
- ✅ `context`: NavigationDriver で使用（click_first_card_or_link で BrowserContext が必要）

**結論**: 未使用フィールドなし

## 追加したコメント

以下のメソッドに「TODO: 削除可能」コメントを追加しました：

1. `_collect_pdp_links` - 薄いラッパーとして残している
2. `_ensure_plp_materialized` - フォールバックとして残している
3. `_force_plp_recover` - フォールバックとして残している
4. `_plp_header_search_fallback` - フォールバックとして残している
5. `_click_first_card_or_link` - フォールバックとして残している

## 次のステップ

Stage 3B/3C/4 で以下を検討：

1. NavigationDriver が常に使われるようになったら、上記のフォールバックメソッドを削除
2. `_collect_pdp_links` の呼び出しをすべて `NavigationDriver.collect_pdp_links` に置き換え
3. 不要になったメソッドの削除

## まとめ

- ✅ 死んでる private メソッドはなし（すべてフォールバックとして使用されている）
- ✅ `_collect_pdp_links` は薄いラッパーとして正しく実装されている
- ✅ NavigationOutcome / NavigationContext のフィールドはすべて使用されている
- ✅ 将来の削除判断がしやすいように TODO コメントを追加

Stage 3A-2 のお片付けは完了しました。

