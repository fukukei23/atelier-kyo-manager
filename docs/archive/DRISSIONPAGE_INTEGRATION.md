# MONCLER専用 Bot対策突破モジュール (DrissionPage) 統合

## 概要

MONCLER のような高度なBot対策（Cloudflare/Akamai）を持つサイトに対して、DrissionPage という Bot 検知に強いライブラリを使用する専用ハンドラを実装しました。

## アーキテクチャ

```
BrowserUseAgent.run()
  ├─ MONCLER 判定
  │   └─ MONCLER_OFFICIAL かつ MonclerDrissionHandler が利用可能
  │       └─ asyncio.to_thread() で MonclerDrissionHandler.run() を実行
  │           └─ 成功 → DiscoveryResult を返す
  │           └─ 失敗 → 警告ログを出して Playwright ルートにフォールバック
  └─ その他のサイト / DrissionPage 未導入 / エラー時
      └─ 既存の Playwright ルート（変更なし）
```

## 実装ファイル

### 1. `app/specialized/moncler_handler.py`

MONCLER 専用の DrissionPage ハンドラ。同期関数として実装されています。

**主な機能:**
- `_start_browser()`: ブラウザ起動（user_data_path でプロファイルを維持）
- `_navigate_to_plp()`: PLP への遷移（検索または直接 URL）
- `_extract_products()`: PLP から商品情報を抽出（最大5件）
- `_extract_product_info()`: 個別商品の情報抽出（タイトル、価格、URL、画像）
- `_save_screenshot()`: スクリーンショット保存
- `_close_browser()`: ブラウザ終了

### 2. `app/agents/browser_use_agent.py`

既存の `run()` メソッドの冒頭に MONCLER 専用分岐を追加。

**変更点:**
- `MonclerDrissionHandler` のインポート（try-except で安全に）
- `run()` メソッドの冒頭に分岐ロジックを追加
- エラー時は自動的に Playwright ルートにフォールバック

## インストール方法

### 1. DrissionPage のインストール

```bash
pip install DrissionPage
```

### 2. Windows 環境

DrissionPage はローカルの Chrome / Chromium を前提とするため、Windows 環境で動作します。

**必要なもの:**
- Chrome または Chromium がインストールされていること
- Python 3.7 以上

## 使い方

### 基本的な使い方

既存の `BrowserUseAgent` の使い方は変わりません。MONCLER の場合のみ自動的に DrissionPage ルートが使用されます。

```python
from app.agents.browser_use_agent import BrowserUseAgent
from app.core.run_context import RunContext

agent = BrowserUseAgent(runtime_kwargs={})
run_context = RunContext()

result = await agent.run(
    site="MONCLER_OFFICIAL",
    query="down jacket",
    site_config=site_config,
    run_context=run_context,
    target_url="https://www.moncler.com/en-int/women/outerwear/all-down-jackets/",
    likely_plp=True,
)
```

### user_data_path のカスタマイズ

`runtime_kwargs` で `user_data_path` を指定できます。

```python
agent = BrowserUseAgent(
    runtime_kwargs={
        "user_data_path": "custom/moncler_profile",
    }
)
```

### ヘッドレスモードの設定

```python
agent = BrowserUseAgent(
    runtime_kwargs={
        "headless": True,  # False にするとブラウザが表示される（デバッグ用）
    }
)
```

## 動作の流れ

1. **ブラウザ起動**
   - `ChromiumPage` を初期化
   - `user_data_path` でプロファイルディレクトリを指定（Cookie・履歴を維持）

2. **PLP への遷移**
   - `target_url` が指定されている場合: その URL に直接アクセス
   - 指定されていない場合: トップページ → クッキー同意 → 検索窓に query を入力 → PLP に遷移

3. **商品情報の抽出**
   - `site_config["selectors"]["plp"]["pdp_link_selectors"]` を使用して商品リンクを取得
   - 各商品からタイトル、価格、URL、画像を抽出（最大5件）

4. **結果の返却**
   - `DiscoveryResult` 形式で返却
   - `evidence["extracted_data"]` に商品情報のリストが含まれる

## フォールバック動作

以下の場合は自動的に Playwright ルートにフォールバックします：

- DrissionPage がインストールされていない場合
- `MonclerDrissionHandler` の初期化に失敗した場合
- DrissionPage ルートの実行中にエラーが発生した場合
- 商品が見つからなかった場合（警告ログを出してフォールバック）

## エラーハンドリング

- エラー時は `DiscoveryResult.ok = False` で返却
- `DiscoveryResult.message` にエラー内容が記録される
- `run_context` に `drission_fallback_error.json` が保存される（エラー時）
- スクリーンショットが保存される（エラー時は `screenshot_drission_error.png`）

## 設定

### site_config の設定例

`app/config/sites/overrides.local.json` の `MONCLER_OFFICIAL` セクションで以下を設定できます：

```json
{
  "MONCLER_OFFICIAL": {
    "base_url": "https://www.moncler.com",
    "navigation": {
      "header_search": {
        "search_input_selector": "input[name='q'], input[type='search']",
        "submit_selector": "button[type='submit']",
        "clear_before_type": true
      }
    },
    "selectors": {
      "plp": {
        "pdp_link_selectors": [
          "div[data-test='product-card'] a",
          "li.product-grid__item a.product-tile__link",
          ".product-cell a"
        ]
      }
    }
  }
}
```

## 既存コードへの影響

**既存の Playwright コードは一切変更していません。**

- 既存の `BrowserUseAgent` の Playwright ルートはそのまま維持
- 既存の `MonclerPLPStrategy` や `moncler_plp_recovery` はそのまま動作
- MONCLER 以外のサイトは従来通り Playwright を使用

## トラブルシューティング

### DrissionPage がインストールされていない場合

```
ImportError: DrissionPage がインストールされていません。
```

**解決方法:**
```bash
pip install DrissionPage
```

### ブラウザが起動しない場合

Chrome または Chromium がインストールされていることを確認してください。

### 商品が見つからない場合

`site_config["selectors"]["plp"]["pdp_link_selectors"]` が正しいセレクタを指定しているか確認してください。

## 今後の改善点

- 取得商品数の調整（現在は5件）
- より詳細な商品情報の抽出
- エラーハンドリングの強化
- パフォーマンスの最適化

## 参考リンク

- [DrissionPage 公式ドキュメント](https://drissionpage.cn/)
- [DrissionPage GitHub](https://github.com/g1879/DrissionPage)

