# Currency コルーチン問題の最終修正案

**作成日時**: 2025-11-30

---

## 問題の根本原因

`currency`フィールドがコルーチンオブジェクトになっている原因は、`get_attribute`のモック設定にあります。

`side_effect`を使っても、まだコルーチンが返されています。これは、`AsyncMock`がコルーチンを返している可能性があります。

---

## 最終的な解決策

`get_attribute`を直接`AsyncMock(return_value="EUR")`として設定し、`side_effect`を使わない方法を試します。

しかし、これでも動作しない場合は、`_extract_currency`メソッド自体がコルーチンを返している可能性があります。

実際のコードを見ると：
```python
content = await locator.get_attribute("content")
currency = (content or "").strip()
if currency:
    return currency
```

これは正しく`await`されているはずです。

問題は、`get_attribute`のモックがコルーチンを返している可能性があります。

---

## 次の試み

`get_attribute`を直接非同期関数として設定します：

```python
currency_locator_first.get_attribute = get_currency_attribute  # AsyncMockを使わない
```

これで、コルーチンが返されないはずです。

---

**ステータス**: 🔍 原因分析中、最終修正案準備中

