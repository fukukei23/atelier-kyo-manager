# Cursor への指示: NexusCore テスト結果自動保存機能の実装

## 指示内容

atelier-kyo-manager で実装済みのテスト結果自動保存機能を、NexusCore プロジェクトにも同様に実装してください。

## 実装する機能

pytest 実行時に、テスト結果を自動的に `docs/reports/TEST_RESULTS_YYYYMMDD_HHMMSS.txt` に保存する機能。

## 必要な作業

### 1. `tests/conftest.py` の作成

`tests/` ディレクトリに `conftest.py` ファイルを作成し、以下の機能を実装してください：

- `pytest_configure(config)`: テスト開始時に結果ファイルのパスを設定
- `pytest_runtest_logreport(report)`: 各テストの結果を収集
- `pytest_sessionfinish(session, exitstatus)`: テスト終了時に結果ファイルに書き込み

**参照**: `atelier-kyo-manager/tests/conftest.py` の実装をそのまま使用してください。

### 2. `pytest.ini` の作成（オプション）

プロジェクトルートに `pytest.ini` を作成してください。

**参照**: `atelier-kyo-manager/pytest.ini` の内容をそのまま使用してください。

### 3. 既存ファイルへの対応

- 既に `tests/conftest.py` が存在する場合は、既存の内容を保持しつつ、新しい関数を追加してください
- 既に `pytest.ini` が存在する場合は、既存の設定を保持してください

### 4. 動作確認

実装後、簡単なテストを実行して、以下を確認してください：

1. テストが正常に実行されること
2. `docs/reports/` ディレクトリに `TEST_RESULTS_*.txt` ファイルが生成されること
3. ターミナルに「✅ テスト結果を保存しました」というメッセージが表示されること

## 実装のポイント

- `docs/reports/` ディレクトリは自動的に作成されるように実装
- ファイル名はタイムスタンプ付きで、上書きされない
- 失敗したテストの詳細（エラーメッセージ）も含める
- 既存のテストに影響を与えないように実装

## 参考ファイル

- `atelier-kyo-manager/tests/conftest.py`
- `atelier-kyo-manager/pytest.ini`

これらを参考にして、NexusCore プロジェクトに同様の実装を追加してください。

