# Currency コルーチン問題の根本原因分析

**作成日時**: 2025-12-01

---

## 問題の詳細

`currency`フィールドがコルーチンオブジェクトになっています：

```
AssertionError: assert <coroutine object AsyncMockMixin._execute_mock_call> == 'EUR'
RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited
```

---

## 根本原因

`_extract_currency`メソッドは正しく`await`されていますが、`get_attribute`のモックがコルーチンを返している可能性があります。

### 実際のコードの流れ

```python
# app/agents/browser/product_extractor.py
async def _extract_currency(self, page: Page, pdp_config: Dict[str, Any]) -> Optional[str]:
    currency_selectors = pdp_config.get("currency", [])
    
    for selector in currency_selectors:
        try:
            locator = page.locator(selector).first  # ← .first を使用
            if await locator.count() == 0:
                continue
            
            content = await locator.get_attribute("content")  # ← ここで await
            currency = (content or "").strip()
            if currency:
                return currency
        except Exception:
            continue
    
    return None
```

### 問題点

1. `page.locator(selector).first`を使用している
2. `test_product_extractor_currency`では`.first`を使っていないため、直接`currency_locator`を返している
3. Monclerテストでは`.first`を使っているため、2段階のLocator構造が必要
4. `AsyncMock`に`.first`プロパティを設定しても、それが正しく動作していない可能性がある

---

## 解決策の方向性

`currency_locator_base`が`AsyncMock`であるため、`.first`プロパティが正しく動作していない可能性があります。

`MagicMock`を使用するか、別の方法で`.first`を設定する必要があります。

---

**ステータス**: 🔍 根本原因分析中

