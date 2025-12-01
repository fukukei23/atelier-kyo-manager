# Currency コルーチン問題の深い分析

**作成日時**: 2025-11-30

---

## 問題の根本原因

`currency`フィールドがコルーチンオブジェクトになっている原因は、`get_attribute`のモック設定にあります。

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

### 現在のモック設定

```python
currency_locator_first = AsyncMock()  # .first が返す Locator
currency_locator_first.count = AsyncMock(return_value=1)
currency_locator_first.get_attribute = AsyncMock(return_value="EUR")
```

### 問題点

`AsyncMock(return_value="EUR")`は、`await currency_locator_first.get_attribute("content")`が呼ばれたときに`"EUR"`を返すべきですが、実際にはコルーチンオブジェクトを返しています。

---

## 解決策

`get_attribute`を非同期関数として定義するか、`side_effect`を使用します。

---

**ステータス**: 🔍 原因分析中、修正案準備中

