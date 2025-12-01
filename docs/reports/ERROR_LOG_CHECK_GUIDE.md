# エラーログ確認ガイド

**状況**: テストは失敗しているが、エラーメッセージの詳細が確認できない

---

## 現在の状況

- ✅ **テストは実行されている**: ターミナル出力で「1 failed, 13 warnings」と表示
- ❌ **エラーメッセージが不明**: テスト結果ファイルに詳細が記録されていない
- ❓ **`conftest.py` のフック**: テスト結果を正しく記録していない可能性

---

## エラーログを確認する方法

### 方法1: WSL環境に直接ログイン（最も確実）

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

### 方法2: エラーログをファイルに保存

```bash
# WSL環境内で
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate

# エラーログをファイルに保存
pytest tests/test_product_extractor.py::test_product_extractor_moncler_pdp_sample -xvs --tb=long > /tmp/moncler_test_error.log 2>&1

# エラーログを確認
cat /tmp/moncler_test_error.log

# Windows側からアクセス（オプション）
# PowerShell から:
# wsl cat /tmp/moncler_test_error.log
```

### 方法3: 簡潔なエラーメッセージを確認

```bash
# WSL環境内で
pytest tests/test_product_extractor.py::test_product_extractor_moncler_pdp_sample -xvs --tb=short
```

`--tb=short` オプションを使用すると、より簡潔なエラーメッセージが表示されます。

---

## テスト結果ファイルの問題

現在、`conftest.py` の `pytest_runtest_logreport` フックがテスト結果を正しく記録していない可能性があります。

### 確認方法

1. **テスト結果ファイルを確認**:
```bash
cat docs/reports/TEST_RESULTS_*.txt | tail -50
```

2. **最新のテスト結果ファイルを確認**:
```bash
ls -lt docs/reports/TEST_RESULTS_*.txt | head -1
cat $(ls -t docs/reports/TEST_RESULTS_*.txt | head -1)
```

---

## 期待されるエラーメッセージの例

テストが失敗する場合、以下のようなエラーメッセージが表示される可能性があります：

### 例1: アサーションエラー
```
AssertionError: assert 1234.0 == 1234.56
```

### 例2: 属性エラー
```
AttributeError: 'MagicMock' object has no attribute 'first'
```

### 例3: キーエラー
```
KeyError: 'images'
```

---

## 次のステップ

1. **エラーメッセージを確認**: 上記の方法でエラーメッセージを確認してください

2. **エラーメッセージを共有**: エラーメッセージを共有していただければ、原因を特定して修正します

3. **修正の適用**: エラーメッセージに基づいて、必要な修正を適用します

---

**重要**: WSL環境の制約により、`run_terminal_cmd` ツールではターミナル出力を直接確認できません。そのため、**WSL環境に直接ログインしてテストを実行する**ことをお勧めします。

