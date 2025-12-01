# 最新テスト結果サマリー

**実行日時**: 2025-11-30 16:43:30  
**テストファイル**: `tests/test_product_extractor.py::test_product_extractor_moncler_pdp_sample`

---

## テスト結果

### ✅ 成功している部分

1. **タイトル抽出**: ✅ 成功
   - `result.title == "Moncler Test Product"` ✅

2. **価格抽出**: ✅ 成功
   - `result.price == 1234.56` ✅（€1,234.56 が正規化されて float に変換）

3. **説明抽出**: ✅ 成功
   - `result.description == "Test product description for Moncler"` ✅

4. **メタデータ**: ✅ 成功
   - `has_title: True` ✅
   - `has_price: True` ✅
   - `has_currency: True` ✅（ただし、currency フィールド自体がコルーチン）

### ❌ 失敗している部分

**通貨抽出**: ❌ 失敗
- **エラー**: `AssertionError: assert <coroutine object AsyncMockMixin._execute_mock_call> == 'EUR'`
- **原因**: `currency`フィールドがコルーチンオブジェクトになっている
- **詳細**: `get_attribute`がコルーチンを返しており、それが正しく`await`されていない

---

## エラーメッセージ

```
AssertionError: assert <coroutine object AsyncMockMixin._execute_mock_call at 0x73f067376f40> == 'EUR'
+  where <coroutine object AsyncMockMixin._execute_mock_call at 0x73f067376f40> = ProductInfo(title='Moncler Test Product', price=1234.56, currency=<coroutine object AsyncMockMixin._execute_mock_call at 0x73f067376f40>, images=[], sizes=[], colors=[], description='Test product description for Moncler', raw_html_path='instance/runs/20251130_164333_529/pdp_raw.html', url='https://www.moncler.com/en-int/women/outerwear/down-jackets/test-product', brand=None, list_price=1234.56, discount_pct=None, metadata={'extraction_timestamp': 1764488613.5882616, 'url': 'https://www.moncler.com/en-int/women/outerwear/down-jackets/test-product', 'has_title': True, 'has_price': True, 'has_currency': True, 'image_count': 0, 'size_count': 0, 'color_count': 0}).currency
```

```
RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited
```

---

## 問題の原因

`currency_locator_first.get_attribute`が`AsyncMock(return_value="EUR")`として設定されていますが、実際のコードでは`await locator.get_attribute("content")`が呼ばれています。モックが正しく動作していない可能性があります。

---

## 修正が必要な点

1. **モックの設定を見直す**: `currency_locator_first.get_attribute`が正しく`"EUR"`を返すようにする
2. **`_extract_currency`メソッドの確認**: `await locator.get_attribute("content")`が正しく`await`されているか確認

---

## 次のステップ

1. `currency_locator_first.get_attribute`のモック設定を修正
2. テストを再実行して確認

---

**ステータス**: 🔍 問題特定済み、修正が必要

