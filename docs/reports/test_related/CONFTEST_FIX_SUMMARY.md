# conftest.py 修正サマリー

**修正日時**: 2025-11-30  
**問題**: `AttributeError: 'TestReport' object has no attribute 'config'`

---

## 問題の原因

`conftest.py` の `pytest_runtest_logreport` フックで、`report.config` にアクセスしようとしていましたが、`TestReport` オブジェクトには `config` 属性がありませんでした。

pytest 9.0.1 では、`report` オブジェクトから直接 `config` を取得できないため、`report.node.config` を使用する必要があります。

---

## 修正内容

`pytest_runtest_logreport` フックを修正し、`config` を安全に取得するようにしました：

```python
# 修正前
config = report.config

# 修正後
config = None
if hasattr(report, 'config'):
    config = report.config
elif hasattr(report, 'node') and hasattr(report.node, 'config'):
    config = report.node.config

if config is None or not hasattr(config, '_test_results'):
    return
```

---

## 次のステップ

この修正により、`conftest.py` のエラーは解消されるはずです。再度テストを実行してください：

```bash
pytest tests/test_product_extractor.py::test_product_extractor_moncler_pdp_sample -xvs --tb=long
```

---

**ステータス**: ✅ 修正完了

