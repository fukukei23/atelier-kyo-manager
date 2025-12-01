# Currency コルーチン問題の最終修正案

**作成日時**: 2025-11-30

---

## 問題の根本原因

`currency`フィールドがコルーチンオブジェクトになっている原因は、`get_attribute`のモック設定にあります。

`_extract_currency`では：
```python
locator = page.locator(selector).first
content = await locator.get_attribute("content")
```

つまり、`.first`を使った2段階のLocator構造が必要です。

---

## 修正案

`test_product_extractor_currency`では、`.first`を使わないため、直接`currency_locator`を返しています。しかし、Monclerテストでは`.first`を使っているため、2段階の構造が必要です。

問題は、`currency_locator_base.first = currency_locator_first`と設定しているが、実際のコードでは`page.locator(selector).first`が呼ばれているため、`currency_locator_base`が返され、その`.first`プロパティが`currency_locator_first`を返す必要があります。

現在の設定：
```python
currency_locator_base = MagicMock()
currency_locator_base.first = currency_locator_first
```

これは正しいはずですが、`currency_locator_first.get_attribute`がコルーチンを返している可能性があります。

---

## 解決策

`currency_locator_first`を`MagicMock`ではなく、`AsyncMock`として作成し、`get_attribute`を正しく設定します。

---

**ステータス**: 🔍 原因分析中、修正案準備中

