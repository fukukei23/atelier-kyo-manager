# テスト結果管理システム

## 概要

テスト実行結果を自動的に構造化して保存し、レポートを生成するシステムを実装しました。

## 機能

1. **テスト結果の構造化保存**
   - JSON形式でテスト結果を保存（`docs/test_results/test_result_YYYYMMDD_HHMMSS.json`）
   - プログラムで解析・集計が可能

2. **Markdownレポートの自動生成**
   - 人間が読みやすい形式のレポート（`docs/test_results/test_report_YYYYMMDD_HHMMSS.md`）
   - サマリー、失敗したテストの詳細、すべてのテスト一覧を含む

3. **完了レポートの自動生成**
   - テスト実行完了時に完了レポートを自動生成（`docs/completion_reports/TEST_EXECUTION_YYYYMMDD_HHMMSS_COMPLETION_REPORT.md`）
   - プロジェクトルールに従った形式

## 使い方

### 基本的な使い方

```bash
# すべてのテストを実行してレポート生成
python run_all_tests.py
```

これにより以下が自動的に実行されます：

1. テスト実行
2. JSON結果の保存（`docs/test_results/`）
3. Markdownレポートの生成（`docs/test_results/`）
4. 完了レポートの生成（`docs/completion_reports/`）

### 詳細な使い方

```bash
# 直接 test_result_manager を使用
python tools/test_result_manager.py tests/

# 特定のテストファイルのみ実行
python tools/test_result_manager.py tests/test_example.py

# 追加のpytest引数を指定
python tools/test_result_manager.py tests/ --args -x --tb=long
```

## ファイル構造

```
docs/
├── test_results/
│   ├── README.md                    # 説明ファイル
│   ├── test_result_YYYYMMDD_HHMMSS.json    # JSON結果
│   └── test_report_YYYYMMDD_HHMMSS.md      # Markdownレポート
└── completion_reports/
    └── TEST_EXECUTION_YYYYMMDD_HHMMSS_COMPLETION_REPORT.md  # 完了レポート
```

## 実装ファイル

### 1. `tools/test_result_manager.py`

テスト結果の管理とレポート生成を行うメインモジュール。

**主要クラス:**
- `TestResult`: 個別のテスト結果
- `TestSessionResult`: テストセッション全体の結果
- `TestResultParser`: pytest の出力を解析
- `TestResultManager`: 結果の保存とレポート生成

### 2. `tools/generate_test_completion_report.py`

テスト完了レポートを生成するスクリプト。

### 3. `run_all_tests.py`

テスト実行とレポート生成を統合したスクリプト。

## JSON結果の構造

```json
{
  "timestamp": "2025-11-28T12:34:56.789012",
  "command": "python -m pytest tests/ -v --tb=short",
  "return_code": 0,
  "total_tests": 10,
  "passed": 8,
  "failed": 1,
  "skipped": 1,
  "errors": 0,
  "duration": 2.34,
  "tests": [
    {
      "name": "test_example",
      "status": "passed",
      "duration": 0.123,
      "file_path": "tests/test_example.py",
      "line_number": null,
      "error_message": null,
      "error_traceback": null
    }
  ],
  "summary": {
    "total": 10,
    "passed": 8,
    "failed": 1,
    "skipped": 1,
    "errors": 0,
    "duration": 2.34,
    "success_rate": 80.0
  }
}
```

## Markdownレポートの内容

- 実行日時・コマンド・終了コード
- サマリー（総テスト数、成功数、失敗数など）
- 失敗したテストの詳細（エラーメッセージ、トレースバック）
- すべてのテスト一覧（ステータス、実行時間）

## 完了レポートの内容

プロジェクトルール（`.cursorrules`）に従った形式：

- 実装日時
- 概要
- テスト実行結果
- 変更ファイル一覧
- 動作確認結果
- 設計上の改善点
- 既知の制約・注意事項
- 次のステップ

## トラブルシューティング

### テスト結果が解析できない場合

pytest の出力形式が変更された可能性があります。`tools/test_result_manager.py` の `TestResultParser` クラスを確認してください。

### レポートが生成されない場合

- `docs/test_results/` ディレクトリが存在することを確認
- `docs/completion_reports/` ディレクトリが存在することを確認
- ファイルの書き込み権限を確認

## 今後の改善点

1. **テスト結果の履歴管理**
   - 過去のテスト結果を保持し、トレンドを分析
   - 成功率の推移をグラフ化

2. **CI/CD統合**
   - GitHub Actions などとの統合
   - 自動的にレポートをアップロード

3. **メール通知**
   - テスト失敗時にメールで通知

## 参考リンク

- [pytest 公式ドキュメント](https://docs.pytest.org/)
- [プロジェクト完了レポートルール](.cursorrules)

