# CR-ATELIER-002: Moncler PLP DOM 構造分析メモ

- **Status:** Analysis Notes
- **Author:** [AI Assistant]
- **Date:** 2025-12-08
- **Related CR:** CR-ATELIER-002_MONCLER_PLP_PDP_EXTRACTION_FIX.md

## 1. 分析対象

### 1.1 分析したDOMファイル

- `instance/runs/20251208_150906_778/failure_dom.html`（最新の実ブラウザ検証結果）
- その他のMoncler実行結果（必要に応じて追加）

### 1.2 分析の目的

- PLPページとSearchページのDOM構造の違いを把握
- PDPリンクがどの階層の要素にぶら下がっているかを特定
- `/products/` 以外のproduct URLパターンが存在するかを確認

## 2. DOM構造の仮説（実ブラウザ検証ベース）

### 2.1 PLPページの構造（仮説）

**メインコンテナ**:
- `<main role="main">` または `<div data-component="ProductListing">`
- セレクタ候補: `main[role='main']`, `div[data-component='ProductListing']`, `section[role='region']`

**商品カード（Product Tile）**:
- `<article data-component="ProductCard">` または `<div data-testid="product-card">`
- セレクタ候補:
  - `article[data-component*='ProductCard']`
  - `[data-testid='product-card']`
  - `[data-testid='product-tile']`
  - `div[class*='product-card' i]`
  - `div[class*='product-tile' i]`

**PDPリンク**:
- 各商品カード内の `<a href="/en-int/.../products/...">`
- セレクタ候補:
  - `article[data-component*='ProductCard'] a[href*='/products/']`
  - `[data-testid*='product-card'] a[href*='/products/']`
  - `a[href*='/en-int/products/']`
  - `a[href*='/products/']`

### 2.2 Searchページの構造（仮説）

**特徴**:
- URLパターン: `/en-int/search` または `/en-lt/en-int/search`
- DOM構造はPLPと類似している可能性がある
- ただし、検索結果のノイズ（関連商品、広告など）が含まれる可能性がある

**PLP相当として扱う条件**:
- DOM上にproduct tileが並んでいる
- `/products/` を含むリンクが一定数存在する
- 明らかな検索トップページ（検索ボックスのみ）ではない

### 2.3 Trapページの構造

**404ページ**:
- `<h1>` に "It's not here" が含まれる
- 商品リストが存在しない

**ロケールゲート**:
- 「Select your location」モーダルが表示されている
- 商品リストが存在しない

**二重ロケールパターン**:
- URLパターン: `/en-lt/en-int/...` または `/en-de/en-int/...`
- 検索ページにリダイレクトされる可能性がある

## 3. セレクタ戦略のレイヤリング案

### 3.1 Primary Layer（site_config準拠）

**セレクタ**:
- `site_config.selectors.plp.pdp_link_selectors` から取得
- `/products/` パターンを前提としたセレクタ

**優先度**: 最高（site_configが正とされる）

### 3.2 Secondary Layer（DOM構造ベース）

**セレクタ**:
- `article[data-component*='ProductCard'] a[href*='/products/']`
- `[data-testid*='product-card'] a[href*='/products/']`
- `div[class*='product-card' i] a[href*='/products/']`

**優先度**: 中（Primaryが失敗した場合に使用）

### 3.3 Tertiary Layer（汎用フォールバック）

**セレクタ**:
- `div:has(a[href*='/products/'])`
- `a[href*='/products/']`（全ページスイープ）

**優先度**: 低（Primary/Secondaryが失敗した場合に使用）

### 3.4 レイヤリングの実装方針

1. **Primary Layer を優先的に使用**
2. **Primary で `raw=0` の場合、Secondary にフォールバック**
3. **Secondary でも `raw=0` の場合、Tertiary にフォールバック**
4. **各レイヤで何件ヒットしたかを Telemetry に記録**

## 4. URLパターンの分析

### 4.1 有効なPDP URLパターン

- `/en-int/women/outerwear/all-down-jackets/products/xxx`
- `/en-int/products/xxx`
- `/en-int/.../products/...`

### 4.2 無効なURLパターン

- `/en-lt/en-int/...`（二重ロケール）
- `/en-int/search`（検索ページ、ただし条件付きで許容）
- `/en-int/404`（404ページ）
- `/en-int/client-service`（サポートページ）

### 4.3 `/products/` 以外のパターン

**現時点での仮説**:
- Monclerは `/products/` パターンのみを使用している可能性が高い
- `/p/` や `/product/` などのパターンは存在しない可能性がある

**確認方法**:
- 実DOMを分析して、実際のURLパターンを確認
- 見つかった場合は、Tertiary Layerに追加

## 5. リダイレクト挙動の分析

### 5.1 確認されたリダイレクトパターン

1. **Locale補正後の再リダイレクト**:
   - `/en-int/...` に補正したが、再び `/en-lt/en-int/...` にリダイレクトされる
   - サーバ側のリダイレクトロジックによる可能性がある

2. **Searchページへのリダイレクト**:
   - `/en-int/...` から `/en-lt/en-int/search` にリダイレクトされる
   - ロケール不一致が原因の可能性がある

### 5.2 防御策の案

1. **再リダイレクトの検出**:
   - `goto` 後にURLを再チェック
   - 二重ロケールパターンが再発した場合、再修正を試みる

2. **Searchページの扱い**:
   - DOM上にproduct tileが並んでいる場合は、PLP相当として扱う
   - ただし、ノイズの多い検索結果は除外する

## 6. 次のステップ

1. **実DOMの詳細分析**:
   - `failure_dom.html` を実際に開いて、DOM構造を確認
   - セレクタ候補を実際のDOMに基づいて更新

2. **セレクタレイヤリングの実装**:
   - Primary / Secondary / Tertiary の順で抽出を試みる
   - 各レイヤで何件ヒットしたかを Telemetry に記録

3. **Self-Healing連携の設計**:
   - `raw=0` が連続した場合、`selector_discovery_agent` にタスクを渡す
   - Telemetry に蓄積した情報を活用して、セレクタを再学習する

