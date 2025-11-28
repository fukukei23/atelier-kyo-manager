# PDP リンク収集のバグ修正

## 問題

実ブラウザテストで以下の問題が発生していました：

1. **`collect_pdp_links` が常に空のリストを返す**
   - `navigation_driver.py` の399行目で、リンクが見つかっても見つからなくても、常に `return []` が実行されていました
   - これにより、Phase 3のノイズフィルタリングと保存処理がスキップされていました

2. **`pdp_link_selectors` が Moncler サイトの構造と一致していない**
   - `MonclerPLPStrategy` では `"a[href*='/products/']"` が見つかっているのに、`collect_pdp_links` では見つかっていませんでした
   - `overrides.local.json` の `pdp_link_selectors` に、実際のMonclerサイトで使用されているセレクタが含まれていませんでした

## 修正内容

### 1. `navigation_driver.py` のバグ修正

**修正前:**
```python
links = sorted(list(found_links))
if not links:
    logger.warning("[PLP→PDP] No PDP hrefs found after all phases.")
return []  # ← 常に実行される（バグ）

# Phase 3: Noise Filtering & Saving
```

**修正後:**
```python
links = sorted(list(found_links))
if not links:
    logger.warning("[PLP→PDP] No PDP hrefs found after all phases.")
    return []  # ← if ブロック内に移動

# Phase 3: Noise Filtering & Saving
```

### 2. `overrides.local.json` の `pdp_link_selectors` を更新

**追加したセレクタ:**
- `"a[href*='/products/']"` - MonclerPLPStrategy で見つかっているセレクタ
- `"a[href*='/product/']"` - バリエーション
- `"a[href*='/p-']"` - 短縮URL形式
- `"[data-qa='product-tile'] a"` - データ属性ベース
- `"main [data-qa='product-tile'] a"` - スコープ付き

**更新後の `pdp_link_selectors`:**
```json
"pdp_link_selectors": [
  "a[href*='/products/']",
  "a[href*='/product/']",
  "a[href*='/p-']",
  "div[data-test='product-card'] a",
  "li.product-grid__item a.product-tile__link",
  ".product-cell a",
  "[data-qa='product-tile'] a",
  "main [data-qa='product-tile'] a"
]
```

## 期待される動作

修正後、以下の動作が期待されます：

1. **Phase 1a/1b でリンクが見つかる**
   - `"a[href*='/products/']"` セレクタで Moncler サイトの PDP リンクが正しく収集される

2. **Phase 3 のノイズフィルタリングが実行される**
   - 収集されたリンクからノイズ（collections, login, cart など）が除去される
   - クリーンな PDP リンクリストが返される

3. **Telemetry への保存が実行される**
   - `raw_pdp_links_v85.5.json` と `raw_hrefs_final_cleaned` が保存される

## 次のステップ

実ブラウザテストを再実行して、PDP リンクが正しく収集されることを確認してください。

