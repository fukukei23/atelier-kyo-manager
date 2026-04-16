# テスト結果

このディレクトリには、テスト実行結果が保存されます。

## ファイル形式

### JSON形式 (`test_result_YYYYMMDD_HHMMSS.json`)

テスト結果の構造化データ。以下の情報を含みます：

- 実行日時
- 実行コマンド
- 終了コード
- サマリー（総テスト数、成功数、失敗数など）
- 個別テストの詳細（名前、ステータス、実行時間、エラーメッセージなど）

### Markdownレポート (`test_report_YYYYMMDD_HHMMSS.md`)

人間が読みやすい形式のレポート。以下のセクションを含みます：

- サマリー
- 失敗したテストの詳細
- すべてのテスト一覧

## 使い方

### テスト実行とレポート生成

```bash
# すべてのテストを実行してレポート生成
python run_all_tests.py

# または、直接 test_result_manager を使用
python tools/test_result_manager.py tests/

# 特定のテストファイルのみ実行
python tools/test_result_manager.py tests/test_example.py
```

### 結果の確認

```bash
# 最新のレポートを確認
ls -lt docs/test_results/test_report_*.md | head -1

# JSON結果を確認
ls -lt docs/test_results/test_result_*.json | head -1
```

## ファイル命名規則

- **JSON結果**: `test_result_YYYYMMDD_HHMMSS.json`
- **Markdownレポート**: `test_report_YYYYMMDD_HHMMSS.md`

タイムスタンプは実行日時（YYYYMMDD_HHMMSS形式）です。

