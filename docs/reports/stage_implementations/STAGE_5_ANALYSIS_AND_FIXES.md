# Stage 5 実装分析と修正レポート

## 1. 分析結果

### 1.1 不整合箇所の特定

現在の実装を `STAGE_5_COMPLETE_IMPLEMENTATION_REPORT.md` と照合した結果、以下の不整合を特定しました：

#### ✅ 既に実装済み（問題なし）

1. `_get_pdp_config()` - 新スキーマ > 旧スキーマ > デフォルトの優先順位で実装済み
2. `_get_price_rules()` - 新スキーマ > 旧スキーマの優先順位で実装済み
3. すべての抽出メソッドが `pdp_config` と `price_rules` を受け取るシグネチャに統一済み
4. 価格正規化が `price_rules` に完全移行済み
5. 画像 URL 正規化が `_normalize_image_url()` 経由になっている
6. JSON-LD / meta fallback が新スキーマに準拠している
7. BrowserExtractionService が graceful degradation を実装済み

#### ⚠️ 軽微な改善点

1. **`_normalize_image_url()` の引数**: 現在は個別引数（`src`, `page_url`, `base_url`）だが、`pdp_config` を渡す方が一貫性がある（ただし、現在の実装でも動作は問題なし）

2. **`_click_size_to_reveal_price()` 内のデフォルト値**: line 690-692 で `availability_patterns` のデフォルト値がハードコードされているが、これは `_get_pdp_config()` 内で既に設定されているため、重複している

3. **`_extract_from_json_ld_or_meta()` 内のデフォルト値**: line 782-784, 795-796 で JSON-LD パスのデフォルト値がハードコードされているが、これは `_get_pdp_config()` 内で既に設定されているため、重複している

### 1.2 実装の品質評価

- ✅ **後方互換性**: 完全に維持されている
- ✅ **設定の優先順位**: 正しく実装されている
- ✅ **Graceful Degradation**: 実装されている
- ✅ **コードの一貫性**: 高い
- ⚠️ **重複コード**: 軽微な重複がある（デフォルト値の設定）

## 2. 修正方針

### 2.1 必須修正（重複コードの削除）

以下の箇所でデフォルト値の重複を削除：

1. `_click_size_to_reveal_price()`: `availability_patterns` のデフォルト値を `_get_pdp_config()` から取得する形に統一
2. `_extract_from_json_ld_or_meta()`: JSON-LD パスのデフォルト値を `_get_pdp_config()` から取得する形に統一

### 2.2 任意修正（一貫性向上）

`_normalize_image_url()` の引数を `pdp_config` に統一する（ただし、現在の実装でも動作は問題なし）

## 3. 修正内容

### 3.1 `_click_size_to_reveal_price()` の修正

**変更前**:
```python
availability_patterns = pdp_config.get("availability_patterns", [
    "out of stock",
    "在庫なし",
])
```

**変更後**:
```python
availability_patterns = pdp_config.get("availability_patterns", [])
```

（`_get_pdp_config()` で既にデフォルト値が設定されているため）

### 3.2 `_extract_from_json_ld_or_meta()` の修正

**変更前**:
```python
paths = json_ld_cfg.get("paths", {
    "price": ["offers.price", "offers[0].price"],
    "currency": ["offers.priceCurrency", "offers[0].priceCurrency"],
})
...
price_paths = paths.get("price", ["offers.price", "offers[0].price"])
currency_paths = paths.get("currency", ["offers.priceCurrency", "offers[0].priceCurrency"])
```

**変更後**:
```python
paths = json_ld_cfg.get("paths", {})
...
price_paths = paths.get("price", [])
currency_paths = paths.get("currency", [])
```

（`_get_pdp_config()` で既にデフォルト値が設定されているため）

## 4. テストの確認

既存のテストは以下の観点で確認済み：

- ✅ Full extraction
- ✅ Partial selectors
- ✅ Price normalization
- ✅ Missing config graceful degradation
- ✅ Metadata counts
- ✅ Config getter (新スキーマ)
- ✅ Config getter (デフォルト)
- ✅ Price rules (新スキーマ)
- ✅ Price rules (旧スキーマ)

すべてのテストが Stage 5 の仕様に準拠しています。

## 5. 結論

現在の実装は Stage 5 の仕様にほぼ完全に準拠しています。軽微な重複コードの削除のみが必要です。

