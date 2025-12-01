# エラーログはファイルで出ないの？- 答え

**作成日時**: 2025-11-30

---

## ✅ エラーログは既にファイルに自動保存されています

### 保存場所

**`docs/reports/TEST_RESULTS_YYYYMMDD_HHMMSS.txt`**

テスト実行のたびに、新しいファイルが自動生成されます。

---

## ファイルの確認方法

### 1. 最新のテスト結果ファイルを確認

```bash
# WSL環境で実行
cd /home/yn441611/atelier-kyo-manager

# 最新のファイルを確認
ls -lt docs/reports/TEST_RESULTS_*.txt | head -1

# ファイルの内容を表示
cat $(ls -t docs/reports/TEST_RESULTS_*.txt | head -1)
```

### 2. Windows のエクスプローラーで確認

以下のパスを開いてください：

```
\\wsl.localhost\Ubuntu\home\yn441611\atelier-kyo-manager\docs\reports\
```

このディレクトリ内の `TEST_RESULTS_*.txt` ファイルをダブルクリックして開けます。

---

## ファイルに含まれる内容

- ✅ 実行日時・終了日時・実行時間
- ✅ テスト数（収集されたテスト数、実行されたテスト数）
- ✅ 結果（成功・失敗・スキップの数）
- ✅ **失敗したテストの詳細**（エラーメッセージとスタックトレース）

---

## なぜターミナルに表示されないのか？

Cursor AIのツール（`run_terminal_cmd`）でコマンドを実行した場合、**ターミナル出力が表示されない制約**があります。これはWSLの問題ではなく、Cursor AIツールの制約です。

### 解決方法

#### 方法1: ファイルで確認（現在の方法）

```bash
# 最新のテスト結果ファイルを確認
cat $(ls -t docs/reports/TEST_RESULTS_*.txt | head -1)
```

#### 方法2: WSL環境に直接ログイン（推奨）

WSL環境に直接ログインして実行すると、**ターミナルに詳細なエラーメッセージが直接表示されます**：

1. Windows PowerShellまたはWindows Terminalを開く
2. `wsl`コマンドを実行してWSLにログイン
3. 以下を実行：

```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate
pytest tests/test_product_extractor.py::test_product_extractor_moncler_pdp_sample -xvs --tb=long
```

これで、**ターミナルに詳細なエラーメッセージが直接表示されます**。

#### 方法3: ログをファイルにリダイレクト

```bash
# テスト実行と同時にログをファイルに保存
pytest tests/test_product_extractor.py::test_product_extractor_moncler_pdp_sample \
  -xvs --tb=long > /tmp/pytest_error.log 2>&1

# エラーログを確認
cat /tmp/pytest_error.log
```

---

## まとめ

- ✅ **エラーログは既にファイルに自動保存されている**
- 📁 **ファイルの場所**: `docs/reports/TEST_RESULTS_*.txt`
- 🔍 **最新ファイルの確認**: `ls -t docs/reports/TEST_RESULTS_*.txt | head -1`
- 💡 **より詳細なエラーログが必要な場合**: WSL環境で直接実行するか、`pytest ... > error.log 2>&1`でリダイレクト

---

**ステータス**: ✅ エラーログは既にファイルに保存されている

