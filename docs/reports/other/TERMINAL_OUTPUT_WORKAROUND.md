# ターミナル出力表示の回避方法

## 問題

Cursor AI の `run_terminal_cmd` ツールでは、コマンドの標準出力が表示されない制約があります。これは Cursor AI のツールの制約であり、WSL環境の問題ではありません。

## 回避方法

### 方法1: ヘルパースクリプトを使用（推奨）

`tools/run_test_with_output.py` を使用することで、テスト結果を表示できます：

```bash
# 特定のテストファイルを実行
python tools/run_test_with_output.py tests/test_product_extractor.py

# 特定のテスト関数を実行
python tools/run_test_with_output.py tests/test_product_extractor.py::test_product_extractor_title

# 詳細な出力を表示
python tools/run_test_with_output.py tests/test_product_extractor.py -v
```

このスクリプトは：
- テストを実行し、出力をリアルタイムで表示
- 最新のテスト結果ファイルのパスを表示
- 終了コードを表示

### 方法2: テスト結果ファイルを確認

`conftest.py` により、テスト結果は自動的に `docs/reports/TEST_RESULTS_YYYYMMDD_HHMMSS.txt` に保存されます：

```bash
# 最新のテスト結果ファイルを確認
ls -lt docs/reports/TEST_RESULTS_*.txt | head -1 | xargs cat

# または、特定のファイルを確認
cat docs/reports/TEST_RESULTS_20250128_143025.txt
```

### 方法3: WSL環境に直接ログイン

WSL環境に直接ログインしてコマンドを実行すると、正常に出力が表示されます：

```bash
# Windows PowerShell または CMD から
wsl

# WSL環境内で
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate
python -m pytest tests/test_product_extractor.py -v
```

### 方法4: 出力を一時ファイルに保存

コマンド実行時に出力を一時ファイルに保存し、その後読み取る：

```bash
# コマンドを実行してファイルに保存
python -m pytest tests/test_product_extractor.py -v > /tmp/pytest_output.txt 2>&1

# ファイルの内容を確認
cat /tmp/pytest_output.txt
```

## 推奨される方法

**開発時**: `tools/run_test_with_output.py` を使用（方法1）

**CI/CD環境**: テスト結果ファイルを確認（方法2）

**直接確認が必要な場合**: WSL環境に直接ログイン（方法3）

## 制約について

この制約は **Cursor AI の `run_terminal_cmd` ツールの根本的な制約**です：

- ✅ コマンドは正常に実行されています（exit code 0）
- ✅ テスト結果はファイルに保存されています
- ⚠️ ただし、標準出力がツール経由で表示されない

この制約を完全に回避することは難しいため、上記の回避方法を使用することをお勧めします。

