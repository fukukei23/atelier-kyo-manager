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

現在の設定：
```python
currency_locator_first.get_attribute = AsyncMock(return_value="EUR")
```

これが正しく動作していない可能性があります。

---

## 解決策

`AsyncMock`の`return_value`ではなく、直接値を返すように設定を変更します。

また、`test_product_extractor_currency`では`.first`を使っていないため、構造が異なります。Monclerテストでは`.first`を使っているため、2段階の構造が必要です。

修正案：
- `currency_locator_first.get_attribute`を、直接値を返す`AsyncMock`として設定し直す
- または、`MagicMock`と`AsyncMock`の組み合わせを見直す

---

**ステータス**: 🔍 原因分析中、修正案準備中

