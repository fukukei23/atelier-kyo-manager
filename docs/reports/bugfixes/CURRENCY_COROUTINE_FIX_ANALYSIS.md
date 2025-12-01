# Currency コルーチン問題の詳細分析と修正案

**作成日時**: 2025-11-30

---

## 問題の詳細

エラーメッセージ：
```
AssertionError: assert <coroutine object AsyncMockMixin._execute_mock_call> == 'EUR'
RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited
```

`currency`フィールド自体がコルーチンオブジェクトになっています。

---

## 原因分析

### 1. 実際のコードの流れ

```python
# app/agents/browser/product_extractor.py
async def _extract_currency(self, page: Page, pdp_config: Dict[str, Any]) -> Optional[str]:
    currency_selectors = pdp_config.get("currency", [])
    
    for selector in currency_selectors:
        try:
            locator = page.locator(selector).first  # ← ここで .first を使用
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

### 2. モックの設定（現在）

```python
currency_locator_first = AsyncMock()  # .first が返す Locator
currency_locator_first.count = AsyncMock(return_value=1)
currency_locator_first.get_attribute = AsyncMock(return_value="EUR")

currency_locator_base = MagicMock()  # page.locator(selector) が返す Locator
currency_locator_base.first = currency_locator_first
currency_locator = currency_locator_base
```

### 3. 問題点

`currency_locator_first.get_attribute = AsyncMock(return_value="EUR")`と設定されているが、実際のコードでは`await locator.get_attribute("content")`が呼ばれています。

しかし、`AsyncMock(return_value="EUR")`は、`await`されたときに`"EUR"`を返すべきです。問題は、モックが正しく設定されていない可能性があります。

---

## 解決策

### 方法1: `get_attribute`を直接`return_value`で設定（推奨）

```python
currency_locator_first = AsyncMock()
currency_locator_first.count = AsyncMock(return_value=1)
currency_locator_first.get_attribute = AsyncMock(return_value="EUR")
```

これが正しい設定のはずですが、まだ動作していません。

### 方法2: `side_effect`を使用

```python
currency_locator_first = AsyncMock()
currency_locator_first.count = AsyncMock(return_value=1)

async def get_attribute_side_effect(attr):
    if attr == "content":
        return "EUR"
    return None

currency_locator_first.get_attribute = AsyncMock(side_effect=get_attribute_side_effect)
```

---

## 次のステップ

1. 実際にどのセレクタがマッチしているかをデバッグログで確認
2. `get_attribute`が実際に呼ばれているかを確認
3. モックの設定を再確認

---

**ステータス**: 🔍 原因分析中

