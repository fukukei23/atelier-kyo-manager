# エラーログファイルの場所と確認方法

**作成日時**: 2025-11-30

---

## ✅ エラーログは既にファイルに自動保存されています

### 保存場所

**パス**: `docs/reports/TEST_RESULTS_YYYYMMDD_HHMMSS.txt`

### 最新ファイルの確認方法

#### WSL環境で確認

```bash
# 最新のテスト結果ファイルをリスト表示
ls -lt docs/reports/TEST_RESULTS_*.txt | head -5

# 最新のファイルを確認
cat $(ls -t docs/reports/TEST_RESULTS_*.txt | head -1)

# または
latest_file=$(ls -t docs/reports/TEST_RESULTS_*.txt | head -1)
cat "$latest_file"
```

#### Windows のエクスプローラーで確認

```
\\wsl.localhost\Ubuntu\home\yn441611\atelier-kyo-manager\docs\reports\
```

このディレクトリ内の `TEST_RESULTS_*.txt` ファイルを開いて確認できます。

---

## ファイルに含まれる内容

テスト結果ファイルには以下の情報が含まれます：

1. **実行日時・終了日時・実行時間**
2. **テスト数**: 収集されたテスト数、実行されたテスト数
3. **結果**: 成功・失敗・スキップの数
4. **失敗したテストの詳細**: エラーメッセージとスタックトレース（`longrepr`）

---

## ファイル名の例

- `TEST_RESULTS_20251130_162943.txt` - 2025年11月30日 16:29:43 に実行
- `TEST_RESULTS_20251130_162212.txt` - 2025年11月30日 16:22:12 に実行

---

## より詳細なエラーログを取得する方法

### 方法1: pytest のログを直接ファイルにリダイレクト

```bash
# テスト実行と同時にログをファイルに保存
pytest tests/test_product_extractor.py::test_product_extractor_moncler_pdp_sample \
  -xvs --tb=long > /tmp/pytest_error.log 2>&1

# エラーログを確認
cat /tmp/pytest_error.log
```

### 方法2: WSL環境で直接実行（推奨）

WSL環境に直接ログインして実行すると、**ターミナルに詳細なエラーメッセージが直接表示されます**：

```bash
# WSLにログイン
wsl

# プロジェクトディレクトリに移動
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate

# テスト実行（詳細な出力）
pytest tests/test_product_extractor.py::test_product_extractor_moncler_pdp_sample -xvs --tb=long
```

### 方法3: ターミナル出力とファイル保存を同時に行う

```bash
# ターミナルにも表示し、ファイルにも保存
pytest tests/test_product_extractor.py::test_product_extractor_moncler_pdp_sample \
  -xvs --tb=long | tee /tmp/pytest_full_output.log
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

================================================================================
詳細なログは pytest の出力を参照してください。
このファイル: /home/yn441611/atelier-kyo-manager/docs/reports/TEST_RESULTS_20251130_162943.txt
================================================================================
```

---

## まとめ

- ✅ **エラーログは既に自動保存されている**: `docs/reports/TEST_RESULTS_*.txt`
- 📁 **ファイルの場所**: `\\wsl.localhost\Ubuntu\home\yn441611\atelier-kyo-manager\docs\reports\`
- 🔍 **最新ファイルの確認**: `ls -t docs/reports/TEST_RESULTS_*.txt | head -1`
- 💡 **より詳細なエラーログが必要な場合**: WSL環境で直接実行するか、`pytest ... > error.log 2>&1`でリダイレクト

---

**ステータス**: ✅ エラーログは既にファイルに保存されている

