# テスト実行結果サマリー

## 実行日時
2025-11-27

## 実行コマンド
```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate
python -m pytest tests/
```

## テスト結果（前回実行時）

- **成功**: 7個
- **失敗**: 14個
- **警告**: 12個（`pytest.mark.asyncio` が未登録）

## 主要な問題点

### 1. `pytest-asyncio` がインストールされていない

**影響**: 14個の async テストが実行できない

**エラーメッセージ例**:
```
async def functions are not natively supported.
You need to install a suitable plugin for your async framework, for example:
  - pytest-asyncio
```

**修正方法**:
```bash
pip install pytest-asyncio
```

### 2. テストコードと実装の不一致

#### 問題1: `NavigationDriver.looks_like_trap_or_legal` が存在しない
- **エラー**: `AttributeError: type object 'NavigationDriver' has no attribute 'looks_like_trap_or_legal'`
- **原因**: 実装では `_looks_like_trap_or_legal` は private メソッド
- **修正**: テストコードを実装に合わせて修正、または public メソッドを追加

#### 問題2: `NavigationDriver.__init__` の引数が異なる
- **エラー**: `TypeError: NavigationDriver.__init__() got an unexpected keyword argument 'ensure_plp_materialized'`
- **原因**: テストコードが古い API を想定している
- **修正**: テストコードを現在の実装に合わせて修正

### 3. SyntaxWarning: invalid escape sequence

**ファイル**: `app/agents/browser/ui_helpers.py:213`
```python
page.locator("text=/United\s+Kingdom\s*\|\s*English/i"),
```

**修正**: raw string を使用
```python
page.locator(r"text=/United\s+Kingdom\s*\|\s*English/i"),
```

## テストファイル一覧

- `test_navigation_driver_stage3a2.py` - NavigationDriver のテスト（一部失敗）
- `test_telemetry_service_stage3b.py` - TelemetryService のテスト（一部失敗）
- `test_session_manager.py` - SessionManager のテスト
- `test_app_smoke.py` - Flask アプリのスモークテスト
- `test_crawler_service.py` - CrawlerService のテスト
- `test_e2e_integration.py` - E2E 統合テスト
- `test_llm_controller.py` - LLM Controller のテスト
- `test_orchestrator.py` - Orchestrator のテスト
- `test_rembg.py` - rembg のテスト
- `test_11.py` - その他のテスト

## 推奨される修正手順

1. **`pytest-asyncio` をインストール**
   ```bash
   pip install pytest-asyncio
   ```

2. **`pytest.ini` または `pyproject.toml` に設定を追加**
   ```ini
   [tool.pytest.ini_options]
   asyncio_mode = "auto"
   ```

3. **テストコードを実装に合わせて修正**
   - `test_navigation_driver_stage3a2.py` の `NavigationDriver` API を確認
   - `test_telemetry_service_stage3b.py` の API を確認

4. **SyntaxWarning を修正**
   - `ui_helpers.py:213` の正規表現を raw string に変更

## 次のステップ

1. `pytest-asyncio` をインストールして再実行
2. テストコードを実装に合わせて修正
3. 全テストが通ることを確認

