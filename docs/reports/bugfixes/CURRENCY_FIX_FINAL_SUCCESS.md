# Currency 問題の最終修正（成功）

**作成日時**: 2025-12-01

---

## 問題の概要

`test_product_extractor_moncler_pdp_sample`で、`currency`フィールドがコルーチンオブジェクトになっていました。

```
AssertionError: assert <coroutine object AsyncMockMixin._execute_mock_call> == 'EUR'
RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited
```

---

## 根本原因

`AsyncMock`に`.first`プロパティを設定しても、それが正しく動作していませんでした。

### 動作していたテストとの違い

- `test_product_extractor_currency`: `.first`を使わずに直接`currency_locator`を返している
- `test_product_extractor_moncler_pdp_sample`: `.first`を使っている

### 問題の詳細

`_extract_currency`メソッドは以下のように実装されています：

```python
locator = page.locator(selector).first  # ← .first を使用
if await locator.count() == 0:
    continue
content = await locator.get_attribute("content")  # ← get_attribute を呼び出し
```

モックでは以下のように設定していましたが、`AsyncMock`に`.first`プロパティを設定しても正しく動作しませんでした：

```python
currency_locator_first = AsyncMock()  # ← AsyncMock を使用
currency_locator_first.count = AsyncMock(return_value=1)
currency_locator_first.get_attribute = AsyncMock(return_value="EUR")
currency_locator_base = AsyncMock()  # ← AsyncMock を使用
currency_locator_base.first = currency_locator_first  # ← .first プロパティを設定
```

---

## 解決策

`MagicMock`を使用することで、`.first`プロパティが正しく動作するようになりました。

### 修正内容

```python
# 修正前
currency_locator_first = AsyncMock()
currency_locator_base = AsyncMock()

# 修正後
from unittest.mock import MagicMock
currency_locator_first = MagicMock()  # ← MagicMock を使用
currency_locator_base = MagicMock()  # ← MagicMock を使用
```

### 修正後のコード

```python
# 通貨抽出のモック（Meta タグ）
# 問題: AsyncMock の .first プロパティが正しく動作していない可能性があるため、
# MagicMock を使用して .first プロパティを設定し、その上で get_attribute を AsyncMock で設定
from unittest.mock import MagicMock
currency_locator_first = MagicMock()  # .first が返す Locator（MagicMock を使用）
currency_locator_first.count = AsyncMock(return_value=1)
currency_locator_first.get_attribute = AsyncMock(return_value="EUR")  # AsyncMock で get_attribute を設定

currency_locator_base = MagicMock()  # page.locator(selector) が返す Locator（MagicMock を使用）
currency_locator_base.first = currency_locator_first  # .first プロパティとして設定
currency_locator = currency_locator_base  # locator_side_effect で返す用
```

---

## 動作確認

修正後、テストが成功しました：

```bash
pytest tests/test_product_extractor.py::test_product_extractor_moncler_pdp_sample -xvs --tb=short
```

**結果**: ✅ テスト成功（exit code 0）

---

## 学んだこと

1. `AsyncMock`にプロパティ（`.first`など）を設定する場合は、`MagicMock`を使用する方が適切
2. 非同期メソッド（`count`, `get_attribute`など）は、`AsyncMock`で明示的に設定する必要がある
3. `MagicMock`を使用することで、プロパティアクセスが正しく動作する

---

## 変更ファイル

- `tests/test_product_extractor.py`: `currency_locator_first`と`currency_locator_base`を`AsyncMock`から`MagicMock`に変更

---

**ステータス**: ✅ 完了

