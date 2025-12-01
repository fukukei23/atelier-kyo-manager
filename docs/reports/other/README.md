# テスト結果ファイル

## 概要

このディレクトリには、pytest 実行時に自動生成されるテスト結果ファイルが保存されます。

## 自動保存機能

**pytest を実行すると、自動的にテスト結果がファイルに保存されます。**

### 保存されるファイル

- **ファイル名**: `TEST_RESULTS_YYYYMMDD_HHMMSS.txt`
- **保存場所**: `docs/reports/`
- **例**: `TEST_RESULTS_20250128_143025.txt`

### 保存される内容

1. **実行情報**
   - 実行日時
   - 終了日時
   - 実行時間

2. **テスト統計**
   - 収集されたテスト数
   - 実行されたテスト数
   - 成功数 ✅
   - 失敗数 ❌
   - スキップ数 ⏭️

3. **失敗テストの詳細**
   - 失敗したテストの名前
   - エラーメッセージとトレースバック

## 使い方

通常通り pytest を実行するだけです：

```bash
pytest
pytest tests/test_plp_driver.py
pytest -v
```

実行後、このディレクトリに結果ファイルが自動的に保存されます。

## 実装

- `tests/conftest.py`: pytest フックで自動保存を実装
- `pytest.ini`: pytest 設定ファイル

詳細は `README_AUTO_TEST_RESULTS.md` を参照してください。

