# Moncler テストの Locator モック修正

**修正日時**: 2025-11-30  
**問題**: Moncler テストですべてのフィールドが`None`または空のリストになる

---

## 問題の原因

1. **`conftest.py`のエラー**: `report.config`が存在しない場合に`AttributeError`が発生
2. **モックの設定不備**: `mock_page.locator.side_effect`が正しく機能していない可能性

---

## 修正内容

### 1. `conftest.py`の修正

`pytest_runtest_logreport`フックで、`report.config`が存在しない場合に`report.node.config`を使用するように修正：

```python
config = None
if hasattr(report, 'config'):
    config = report.config
elif hasattr(report, 'node') and hasattr(report.node, 'config'):
    config = report.node.config

if config is None or not hasattr(config, '_test_results'):
    return
```

### 2. `locator_side_effect`の改善

`tests/test_product_extractor.py`の`test_product_extractor_moncler_pdp_sample`テストで、`locator_side_effect`の条件を強化：

- セレクタ文字列のさまざまなバリエーションに対応
- `any([...])`を使用して複数の条件をチェック
- `mock_page.locator = MagicMock(side_effect=locator_side_effect)`として明示的に設定

---

## 次のステップ

テストを実行して、修正が正しく機能するか確認してください：

```bash
pytest tests/test_product_extractor.py::test_product_extractor_moncler_pdp_sample -xvs --tb=long
```

---

**ステータス**: ✅ 修正完了（テスト実行で確認が必要）

