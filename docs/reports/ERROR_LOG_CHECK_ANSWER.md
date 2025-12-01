# エラーログ確認への回答

**質問**: 「エラーログ出てる？」

---

## 答え: ❌ エラーログが確認できません

テスト結果ファイル（`TEST_RESULTS_*.txt`）を見ると、「実行されたテスト数: 0」となっており、エラーメッセージの詳細が記録されていません。

---

## 原因

`conftest.py` の `pytest_runtest_logreport` フックは、`report.when == 'call'` の時のみテスト結果を記録します。しかし、テストが**実行前に失敗**している場合（例: fixture のエラー、インポートエラーなど）は、`report.when` が `'call'` にならないため、結果が記録されません。

---

## エラーログを確認する方法

### 推奨方法: WSL環境に直接ログイン

```bash
# Windows PowerShell または Windows Terminal から
wsl

# WSL環境内で
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate

# テストを実行（エラーメッセージが直接表示される）
pytest tests/test_product_extractor.py::test_product_extractor_moncler_pdp_sample -xvs --tb=long
```

この方法なら、**エラーメッセージが直接ターミナルに表示**されます。

---

## 実行すると表示される情報

ターミナルに直接表示されるエラーメッセージには、以下のような情報が含まれます：

1. **テスト名**: `test_product_extractor_moncler_pdp_sample`
2. **エラーの種類**: `AssertionError`, `AttributeError`, `KeyError` など
3. **エラーメッセージ**: 具体的なエラー内容
4. **スタックトレース**: エラーが発生した場所
5. **失敗したアサーション**: 期待値と実際の値

---

## 次のステップ

1. **エラーメッセージを確認**: 上記の方法でエラーメッセージを確認してください

2. **エラーメッセージを共有**: エラーメッセージを共有していただければ、原因を特定して修正します

3. **修正の適用**: エラーメッセージに基づいて、必要な修正を適用します

---

**結論**: 現在、エラーログを確認できない状態です。WSL環境に直接ログインしてテストを実行すると、エラーメッセージが表示されます。

