# カバレッジ測定レポート

## 測定日時
2025-11-27

## 測定対象
- `app/agents/browser/` 配下の全ファイル
- 成功したテストのみで測定（7個成功、14個失敗）

## 総合カバレッジ

```
総合カバレッジ: 12.0%
カバー行数: 212 / 1,810
未カバー行数: 1,598
```

## 主要ファイルのカバレッジ詳細

| ファイル | カバレッジ | カバー行数 | 総行数 |
|---------|-----------|-----------|--------|
| `navigation_driver.py` | **9%** | 61 / 679 | 618行未カバー |
| `telemetry.py` | **33%** | 53 / 163 | 110行未カバー |
| `extractor.py` | **17%** | 44 / 253 | 209行未カバー |
| `session_manager.py` | **16%** | 54 / 346 | 292行未カバー |
| `ui_helpers.py` | **0%** | 0 / 160 | 160行未カバー |
| `plugin_api.py` | **0%** | 0 / 66 | 66行未カバー |
| `moncler_patch.py` | **0%** | 0 / 80 | 80行未カバー |
| `settings.py` | **0%** | 0 / 52 | 52行未カバー |

## テスト実行状況

- **成功**: 7個
- **失敗**: 14個
- **警告**: 12個（`pytest.mark.asyncio` が未登録）

### 失敗の主な原因

1. **`pytest-asyncio` がインストールされていない**
   - async テストが実行できない
   - 14個のテストがこの理由で失敗

2. **テストコードと実装の不一致**
   - `NavigationDriver.looks_like_trap_or_legal` が存在しない（`_looks_like_trap_or_legal` は private）
   - `NavigationDriver.__init__` の引数がテストと異なる

## 改善提案

### 1. 即座に対応可能

```bash
# pytest-asyncio をインストール
pip install pytest-asyncio
```

### 2. テストコードの修正

- `test_navigation_driver_stage3a2.py` を実装に合わせて修正
- `NavigationDriver` の実際の API に合わせる

### 3. カバレッジ向上の優先順位

1. **`navigation_driver.py` (9%)**: 最重要ファイルだがカバレッジが低い
2. **`telemetry.py` (33%)**: 比較的高いが、まだ改善の余地あり
3. **`extractor.py` (17%)**: PDP 抽出ロジックのテストが必要
4. **`session_manager.py` (16%)**: セッション管理のテストが必要

## 注意事項

- 現在のカバレッジは**成功したテストのみ**で測定されています
- async テストが実行できれば、カバレッジはさらに向上する可能性があります
- `ui_helpers.py`、`plugin_api.py` などはテストが存在しないため 0% です
