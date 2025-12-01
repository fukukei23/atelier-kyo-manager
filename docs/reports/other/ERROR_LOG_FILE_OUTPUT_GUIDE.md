# エラーログのファイル出力について

**更新日時**: 2025-11-30

---

## 現在の状況

エラーログは既にファイルに自動保存されています：

1. **テスト結果ファイル**: `docs/reports/TEST_RESULTS_YYYYMMDD_HHMMSS.txt`
   - テスト実行ごとに自動生成
   - 実行日時、終了コード、失敗したテストの詳細（`longrepr`）を含む

2. **保存タイミング**: `pytest`実行終了時（`conftest.py`の`pytest_sessionfinish`フック）

---

## ファイルの確認方法

### 1. 最新のテスト結果ファイルを確認

```bash
# 最新のファイルを表示
ls -lt docs/reports/TEST_RESULTS_*.txt | head -1

# ファイルの内容を確認
cat $(ls -t docs/reports/TEST_RESULTS_*.txt | head -1)
```

### 2. 特定のテストのエラーログを確認

```bash
# 最新のテスト結果ファイルを確認
latest_file=$(ls -t docs/reports/TEST_RESULTS_*.txt | head -1)
cat "$latest_file"
```

---

## 改善案

### 問題点

現在、`TEST_RESULTS_*.txt`ファイルに「実行されたテスト数: 0」と表示される場合があります。これは`pytest_runtest_logreport`フックが正しく動作していない可能性があります。

### 改善方法

より詳細なエラーログをファイルに出力するため、以下の改善が可能です：

1. **詳細なスタックトレースを含める**: `--tb=long`を使用した場合の詳細なトレースバックをファイルに保存
2. **pytest のログをファイルにリダイレクト**: `pytest ... > test_output.log 2>&1`
3. **カスタムログハンドラー**: `conftest.py`にログハンドラーを追加して、エラーを直接ファイルに書き込む

---

## 推奨される使用方法

### 方法1: テスト結果ファイルを確認（現在の方法）

```bash
# テスト実行後、最新の結果ファイルを確認
ls -lt docs/reports/TEST_RESULTS_*.txt | head -1 | awk '{print $NF}' | xargs cat
```

### 方法2: pytest のログをファイルにリダイレクト

```bash
# テスト実行と同時にログをファイルに保存
pytest tests/test_product_extractor.py::test_product_extractor_moncler_pdp_sample -xvs --tb=long > /tmp/pytest_error.log 2>&1

# エラーログを確認
cat /tmp/pytest_error.log
```

### 方法3: WSL環境で直接実行（最も詳細）

WSL環境に直接ログインして実行すると、ターミナルに詳細なエラーが表示されます：

```bash
# WSLにログイン
wsl

# プロジェクトディレクトリに移動
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate

# テスト実行（詳細な出力）
pytest tests/test_product_extractor.py::test_product_extractor_moncler_pdp_sample -xvs --tb=long
```

---

## 現在のテスト結果ファイルの例

```
================================================================================
テスト実行結果
実行日時: 2025-11-30 16:29:43
終了日時: 2025-11-30 16:29:48
実行時間: 4.86秒
================================================================================

収集されたテスト数: 1
実行されたテスト数: 0
✅ 成功: 0
❌ 失敗: 0
⏭️  スキップ: 0
終了コード: 1

失敗したテスト:
--------------------------------------------------------------------------------
（失敗したテストの詳細がここに表示される）
--------------------------------------------------------------------------------
```

---

## 次のステップ

1. `conftest.py`を改善して、より詳細なエラーログを保存する
2. pytest のログをファイルにリダイレクトするヘルパースクリプトを作成する
3. エラーログを専用のファイル（`docs/reports/ERROR_LOG_*.txt`）に保存する機能を追加する

---

**ステータス**: ✅ エラーログは既にファイルに保存されている（改善の余地あり）

