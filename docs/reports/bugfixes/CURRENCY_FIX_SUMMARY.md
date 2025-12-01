# Currency コルーチン問題の修正まとめ

**作成日時**: 2025-12-01

---

## 問題の根本原因

`currency`フィールドがコルーチンオブジェクトになっていた原因は、`get_attribute`のモック設定方法にありました。

### 問題のある設定（以前）

```python
async def get_currency_attribute(attr: str):
    if attr == "content":
        return "EUR"
    return None
currency_locator_first.get_attribute = AsyncMock(side_effect=get_currency_attribute)
```

この方法では、`AsyncMock`がコルーチンオブジェクトを返していました。

---

## 解決策

`get_attribute`を直接`AsyncMock(return_value="EUR")`として設定しました。これは、動作している`test_product_extractor_currency`と同じパターンです。

### 修正後の設定

```python
currency_locator_first = AsyncMock()  # .first が返す Locator
currency_locator_first.count = AsyncMock(return_value=1)
currency_locator_first.get_attribute = AsyncMock(return_value="EUR")  # side_effect を使わず、直接 return_value を使用

currency_locator_base = AsyncMock()  # page.locator(selector) が返す Locator
currency_locator_base.first = currency_locator_first  # .first プロパティとして設定
currency_locator = currency_locator_base
```

---

## 変更ファイル

- `tests/test_product_extractor.py`
  - `test_product_extractor_moncler_pdp_sample`関数内の`currency_locator_first.get_attribute`の設定を修正

---

## 次のステップ

テストを実行して、修正が正しく動作することを確認してください：

```bash
pytest tests/test_product_extractor.py::test_product_extractor_moncler_pdp_sample -xvs --tb=long
```

---

**ステータス**: ✅ 修正完了

