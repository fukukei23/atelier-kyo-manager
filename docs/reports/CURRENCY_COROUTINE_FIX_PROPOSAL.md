# Currency コルーチンオブジェクト問題の修正案

**作成日時**: 2025-11-30

---

## 問題の原因

`currency`がコルーチンオブジェクトになっている原因は、`currency_locator_first.get_attribute`のモック設定にあります。

現在の設定：
```python
currency_locator_first = AsyncMock()
currency_locator_first.count = AsyncMock(return_value=1)
currency_locator_first.get_attribute = AsyncMock(return_value="EUR")
```

問題：`AsyncMock`として作成された`currency_locator_first`に`get_attribute`を直接設定しても、正しく動作していない可能性があります。

---

## 解決策

### 方法1: `MagicMock`を使用して`get_attribute`を設定

```python
from unittest.mock import AsyncMock, MagicMock

currency_locator_first = MagicMock()
currency_locator_first.count = AsyncMock(return_value=1)
currency_locator_first.get_attribute = AsyncMock(return_value="EUR")
```

### 方法2: `AsyncMock`を`spec`で制約

```python
currency_locator_first = AsyncMock(spec=['count', 'get_attribute', 'inner_text'])
currency_locator_first.count = AsyncMock(return_value=1)
currency_locator_first.get_attribute = AsyncMock(return_value="EUR")
```

### 方法3: `side_effect`を使用

```python
currency_locator_first = AsyncMock()
currency_locator_first.count = AsyncMock(return_value=1)
currency_locator_first.get_attribute = AsyncMock(side_effect=lambda attr: "EUR" if attr == "content" else None)
```

---

## 推奨される修正

他のテスト（`test_product_extractor_currency`など）を見ると、`AsyncMock`を使用していますが、`currency_locator`が直接`AsyncMock`として作成されています。

問題は、`currency_locator_first`が`AsyncMock`として作成されているため、`get_attribute`が正しく動作していない可能性があります。

修正案：
```python
# 通貨抽出のモック（Meta タグ）
currency_locator_first = MagicMock()  # AsyncMock ではなく MagicMock
currency_locator_first.count = AsyncMock(return_value=1)
currency_locator_first.get_attribute = AsyncMock(return_value="EUR")
```

---

**ステータス**: 🔍 原因特定済み、修正案準備中

