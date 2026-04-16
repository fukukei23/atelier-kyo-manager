# Currency コルーチンオブジェクト問題の修正

**作成日時**: 2025-11-30  
**問題**: `currency`が`<coroutine object>`になっている

---

## 問題の原因

`_extract_currency`メソッドでは、`await locator.get_attribute("content")`と正しく`await`されていますが、モックの`get_attribute`がコルーチンオブジェクトを返している可能性があります。

エラーメッセージ：
```
AssertionError: assert <coroutine object AsyncMockMixin._execute_mock_call> == 'EUR'
```

これは、`get_attribute`が`AsyncMock`として設定されているが、実際のコードで正しく`await`されていないか、またはモックの設定に問題があることを示しています。

---

## 確認が必要な点

1. `currency_locator_first.get_attribute`が`AsyncMock(return_value="EUR")`として正しく設定されているか
2. `_extract_currency`が`await locator.get_attribute("content")`を正しく呼び出しているか
3. `ProductInfo`に設定される際に、コルーチンが正しく`await`されているか

---

## 次のステップ

`tests/test_product_extractor.py`の`test_product_extractor_moncler_pdp_sample`テストで、`currency_locator_first.get_attribute`の設定を確認し、必要に応じて修正します。

---

**ステータス**: 🔍 調査中

