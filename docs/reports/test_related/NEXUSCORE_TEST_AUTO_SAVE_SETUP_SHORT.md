# NexusCore: テスト結果自動保存の設定（簡潔版）

## 概要

atelier-kyo-manager で実装したテスト結果自動保存機能を NexusCore にも追加してください。

## 実装手順

### 1. `tests/conftest.py` を作成

`tests/` ディレクトリに `conftest.py` を作成し、以下のコードをコピー＆ペーストしてください。

**参照元**: `atelier-kyo-manager/tests/conftest.py` をそのままコピーしてください。

### 2. `pytest.ini` を作成（オプション）

プロジェクトルートに `pytest.ini` を作成してください。

**参照元**: `atelier-kyo-manager/pytest.ini` をそのままコピーしてください。

### 3. 動作確認

```bash
pytest
```

実行後、`docs/reports/TEST_RESULTS_*.txt` が自動生成されることを確認してください。

## 完了

これで完了です。次回から pytest 実行時に自動的に結果ファイルが生成されます。

---

**詳細版**: `NEXUSCORE_TEST_AUTO_SAVE_SETUP.md` を参照

