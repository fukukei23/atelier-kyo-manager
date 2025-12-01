# エラーログのファイル出力について（完全版）

**作成日時**: 2025-11-30

---

## ✅ エラーログは既にファイルに自動保存されています

### 現在の状況

エラーログは **既に自動保存されています** が、以下の問題があります：

1. **自動保存ファイル**: `docs/reports/TEST_RESULTS_YYYYMMDD_HHMMSS.txt`
   - ✅ テスト実行のたびに自動生成される
   - ⚠️ 場合によっては「実行されたテスト数: 0」と表示される（記録が不完全な場合がある）

2. **ターミナル出力**: Cursor AIのツールで実行した場合、ターミナルに表示されない
   - ⚠️ これはWSLの問題ではなく、Cursor AIツールの制約

---

## エラーログファイルの確認方法

### 方法1: 最新のテスト結果ファイルを確認

```bash
# WSL環境で実行
cd /home/yn441611/atelier-kyo-manager

# 最新のファイルを確認
ls -lt docs/reports/TEST_RESULTS_*.txt | head -1

# ファイルの内容を表示
cat $(ls -t docs/reports/TEST_RESULTS_*.txt | head -1)
```

### 方法2: Windows のエクスプローラーで確認

以下のパスを開いてください：

```
\\wsl.localhost\Ubuntu\home\yn441611\atelier-kyo-manager\docs\reports\
```

このディレクトリ内の `TEST_RESULTS_*.txt` ファイルをダブルクリックして開けます。

---

## より詳細なエラーログを取得する方法

### 推奨方法: WSL環境で直接実行

WSL環境に直接ログインして実行すると、**ターミナルに詳細なエラーメッセージが直接表示されます**：

```bash
# 1. Windows PowerShellまたはWindows Terminalを開く
# 2. WSLにログイン
wsl

# 3. プロジェクトディレクトリに移動
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate

# 4. テスト実行（詳細な出力）
pytest tests/test_product_extractor.py::test_product_extractor_moncler_pdp_sample -xvs --tb=long
```

**これで、ターミナルに詳細なエラーメッセージとスタックトレースが直接表示されます。**

### 代替方法: ログをファイルにリダイレクト

```bash
# テスト実行と同時にログをファイルに保存
pytest tests/test_product_extractor.py::test_product_extractor_moncler_pdp_sample \
  -xvs --tb=long > /tmp/pytest_error.log 2>&1

# エラーログを確認
cat /tmp/pytest_error.log
```

### さらに良い方法: ターミナルにも表示し、ファイルにも保存

```bash
# ターミナルにも表示し、ファイルにも保存
pytest tests/test_product_extractor.py::test_product_extractor_moncler_pdp_sample \
  -xvs --tb=long | tee /tmp/pytest_full_output.log
```

---

## 現在の問題

### 問題1: テスト結果ファイルに「実行されたテスト数: 0」と表示される

これは`pytest_runtest_logreport`フックが正しく動作していない可能性があります。ただし、テスト自体は実行されており、ターミナルにはエラーが表示されています。

### 問題2: ターミナル出力が表示されない

Cursor AIのツール（`run_terminal_cmd`）でコマンドを実行した場合、ターミナル出力が表示されない制約があります。これはWSLの問題ではなく、Cursor AIツールの制約です。

---

## 解決策

### 最も確実な方法: WSL環境で直接実行

1. **Windows PowerShellまたはWindows Terminalを開く**
2. **`wsl`コマンドを実行してWSLにログイン**
3. **プロジェクトディレクトリに移動してテストを実行**

これで、**ターミナルに詳細なエラーメッセージが直接表示されます**。

---

## まとめ

- ✅ **エラーログは既にファイルに自動保存されている**: `docs/reports/TEST_RESULTS_*.txt`
- ⚠️ **ファイルの内容が不完全な場合がある**: 「実行されたテスト数: 0」など
- 💡 **最も詳細なエラーログを取得するには**: **WSL環境で直接実行**することを推奨
- 📁 **ファイルの場所**: `\\wsl.localhost\Ubuntu\home\yn441611\atelier-kyo-manager\docs\reports\`

---

**ステータス**: ✅ エラーログは既にファイルに保存されている（ただし、WSL環境で直接実行する方がより詳細）

