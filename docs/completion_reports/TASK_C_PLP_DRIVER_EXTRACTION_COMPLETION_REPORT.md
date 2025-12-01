# Task C: Stage 4 – 汎用 PLP Driver の切り出し 完了レポート

## 実装日時
2025年11月28日

## 概要

PLP → PDP のナビゲーションロジックを BrowserUseAgent から切り出し、
サイト非依存の「汎用 PLP Driver」モジュール（`app/agents/browser/plp_driver.py`）にまとめました。

## 実装ステップ

### 1. PlpDriver クラスの作成

**新規ファイル**: `app/agents/browser/plp_driver.py`

**主要なクラスとデータ構造**:
- `PlpNavigationResult`: ナビゲーション結果を格納するデータクラス
  - `pdp_url`: PDP URL
  - `pdp_opened_in_new_tab`: 新タブで開かれたかどうか
  - `plp_url`: PLP URL
  - `tiles_seen`: 見つかったタイル数
  - `trap_detected`: Trap ページが検出されたかどうか
  - `trap_reason`: Trap 検出理由
  - `recovery_attempted`: リカバリが試行されたかどうか

- `PlpDriver`: PLP → PDP ナビゲーションを処理する汎用ドライバ
  - `navigate_to_pdp()`: メインのナビゲーションメソッド
  - `_materialize_plp_tiles()`: PLP タイルのマテリアライズ
  - `_click_tile_and_navigate_to_pdp()`: タイルクリック → PDP 遷移
  - `_click_and_capture_navigation()`: 新タブ／同タブ／SPA の URL 変化を待つレース処理
  - `_looks_like_trap_or_legal()`: Trap/Legal ページ検出
  - `_recover_from_trap()`: Trap ページからの復帰
  - `_handle_overlays()`: Cookie バナー、Geo モーダル、オーバーレイ処理

### 2. BrowserUseAgent の修正

**変更ファイル**: `app/agents/browser_use_agent.py`

**主な変更点**:
- `PlpDriver` のインポートを追加
- フォールバック処理（`_click_first_card_or_link`）で `PlpDriver` を使用
- 既存の `_click_first_card_or_link` はフォールバックとして残存（互換性維持）

### 3. 移動したロジック

以下のロジックを BrowserUseAgent から PlpDriver に移動：

1. **PLP タイルのマテリアライズ**
   - `_ensure_plp_materialized()` のロジックを `_materialize_plp_tiles()` に移動
   - スクロール、タイル数カウント、ロケールリダイレクト検出

2. **タイルクリック → PDP 遷移**
   - `_click_first_card_or_link()` のロジックを `_click_tile_and_navigate_to_pdp()` に移動
   - `_click_and_capture_navigation()` のロジックを移動（新タブ／同タブ／SPA のレース処理）

3. **Trap/Legal ページ検出と回避**
   - `_looks_like_trap_or_legal()` を NavigationDriver 経由で呼び出し
   - `_recover_from_trap()` を NavigationDriver 経由で呼び出し

4. **Geo モーダル・Cookie バナー処理**
   - `_accept_cookies_if_present()` を移動
   - `_dismiss_geo_modal()` を移動
   - `_kill_overlays()` を移動

### 4. site_config への依存

PlpDriver は以下の `site_config` キーを参照：

- `selectors.pdp.pdp_link_selectors`: PDP リンクのセレクタ
- `selectors.pdp.plp_container_selectors`: PLP コンテナのセレクタ
- `selectors.pdp.blocklist_href_substrings`: ブロックリスト
- `selectors.ui.cookie_accept`: Cookie バナーのセレクタ
- `navigation.overlays.geo_modal_selectors`: Geo モーダルのセレクタ
- `discovery_settings.plp_scroll_rounds`: スクロール試行回数
- `locale.prefer`: 優先ロケール
- `allowed_domain`: 許可ドメイン

## 変更ファイル一覧

### 新規作成ファイル

1. **`app/agents/browser/plp_driver.py`**
   - PlpDriver クラスと PlpNavigationResult データクラス
   - PLP → PDP ナビゲーションロジック

### 変更ファイル

1. **`app/agents/browser_use_agent.py`**
   - `PlpDriver` のインポートを追加
   - フォールバック処理で `PlpDriver` を使用
   - 既存の `_click_first_card_or_link` はフォールバックとして残存

## 動作確認結果

### 静的解析結果
- リンター警告: Playwright のインポート解決警告（実行時には問題なし）
- 型チェック: 問題なし

### コードレビュー結果
- PlpDriver クラスの実装: ✅ 完了
- BrowserUseAgent の統合: ✅ 完了
- site_config への依存: ✅ 完了
- 既存のフォールバック処理: ✅ 維持

### テスト結果
- ユニットテスト: Task E で実装予定
- 統合テスト: Task E で実装予定

## 設計上の改善点

### アーキテクチャの改善
1. **責務の分離**
   - PLP → PDP ナビゲーションロジックが PlpDriver に集約
   - BrowserUseAgent は PlpDriver を呼び出すだけの薄いハブに

2. **サイト非依存化**
   - サイト固有のロジックは `site_config` から参照
   - コード側は汎用的なナビゲーションロジックのみ

3. **再利用性の向上**
   - PlpDriver は他のエージェントからも使用可能
   - NavigationDriver との役割分担が明確に

### 将来の拡張性への配慮
- 新しいサイトを追加する際、コード改変ではなく `site_config` の追加だけで対応可能
- PlpDriver のメソッドを拡張することで、新しいナビゲーションパターンに対応可能

### コード品質の向上
- テスト容易性の向上（PlpDriver を独立してテスト可能）
- 可読性の向上（BrowserUseAgent が簡潔に）

## 既知の制約・注意事項

### 既存コードとの互換性
- 既存の `_click_first_card_or_link` はフォールバックとして残存
- NavigationDriver との役割分担：
  - NavigationDriver: PLP ナビゲーション、PDP リンク収集
  - PlpDriver: タイルクリック → PDP 遷移（単一タイル）

### 制限事項やトレードオフ
- 現在の実装では、PlpDriver は単一タイルのクリック → PDP 遷移を担当
- 複数 PDP リンクの収集は NavigationDriver が担当
- 将来的には、PlpDriver と NavigationDriver の統合を検討

### 移行時の注意点
- 既存の実行スクリプトは変更不要
- `site_config` に必要なキーが定義されていることを確認

## 次のステップ

### 推奨されるフォローアップアクション

1. **Task E: テストの追加**
   - PlpDriver のユニットテスト（モック Page/HTML）
   - BrowserUseAgent の E2E テスト（Moncler 統合テスト）

2. **PlpDriver と NavigationDriver の統合**
   - 役割分担の明確化
   - 重複ロジックの削減

3. **site_config スキーマの標準化**
   - PlpDriver が使用する `site_config` キーの標準化
   - ドキュメント化

## 関連ファイル

- `app/agents/browser/plp_driver.py` - PlpDriver クラス
- `app/agents/browser_use_agent.py` - BrowserUseAgent クラス
- `app/agents/browser/navigation_driver.py` - NavigationDriver クラス（既存）

