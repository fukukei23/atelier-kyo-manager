# MONCLER_OFFICIAL.json 更新完了レポート

## 更新日時
2025-11-28

## 実施内容

### 1. navigation ブロックへの追加

`app/config/sites/overrides.local.json` の `MONCLER_OFFICIAL.navigation` ブロックに以下を追加：

- `trap_url_patterns`: trap 判定に使用される URL パターン
- `legal_url_patterns`: リーガルページ判定に使用される URL パターン
- `header_search`: ヘッダ検索 fallback の設定
  - `search_input_selector`: 検索入力フィールドのセレクタ
  - `submit_selector`: 検索送信ボタンのセレクタ
  - `clear_before_type`: 入力前にクリアするかどうか
- `overlays`: overlay 除去の設定
  - `cookie_banner_selectors`: cookie バナーのセレクタ
  - `geo_modal_selectors`: geo モーダルのセレクタ
  - `generic_close_buttons`: 汎用閉じるボタンのセレクタ
- `fallback.click_first_card`: カードクリック fallback の設定
  - `enabled`: 有効/無効
  - `card_selectors`: カードのセレクタ
- `plp`: PLP materialize の設定
  - `supports_header_search`: ヘッダ検索のサポート
  - `supports_card_click_fallback`: カードクリック fallback のサポート
  - `supports_locale_normalization`: ロケール正規化のサポート
  - `min_pdp_links`: 最小 PDP リンク数
  - `max_scroll_iterations`: 最大スクロール回数
  - `scroll_pause_ms`: スクロール間の待機時間
  - `plp_timeout_ms`: PLP タイムアウト

### 2. selectors.plp の追加

`app/config/sites/overrides.local.json` の `MONCLER_OFFICIAL.selectors` ブロックに `plp` を追加：

- `container_selectors`: PLP コンテナのセレクタ
- `card_selectors`: 商品カードのセレクタ
- `pdp_link_selectors`: PDP リンクのセレクタ
- `price_selectors`: 価格のセレクタ
- `visible_price_selectors`: 表示価格のセレクタ

## 確認方法

### 1. JSON構文チェック

```bash
python3 -m json.tool app/config/sites/overrides.local.json
```

### 2. site_config 接続テスト

```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate
python test_site_config_connection.py
```

または

```bash
./verify_site_config_update.sh
```

### 3. 期待される結果

以下の警告が消えていることを確認：

- ✅ `selectors.plp` - 見つかりました
- ✅ `navigation.header_search` - 見つかりました
- ✅ `navigation.overlays` - 見つかりました

### 4. 実ブラウザテスト（オプション）

```bash
python run_orchestrator.py \
  --site "MONCLER_OFFICIAL" \
  --query "down jacket" \
  --target_url "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB" \
  --likely_plp true \
  --auto-heal \
  --headful
```

確認ポイント：
- PLP → PDP に正常に遷移する
- コンソールログに `[NavigationDriver]` 系のログが出る
- `selectors.plp` / `navigation.header_search` を使用しているメッセージが出る

## 次のステップ

1. **テストの再実行**: `verify_site_config_update.sh` を実行して、警告が消えていることを確認
2. **実ブラウザテスト**: 余裕があれば Moncler 実ブラウザテストを実行
3. **他のサイトへの展開**: 他のサイト（GIVENCHY_OFFICIAL など）にも同様の構造を追加

