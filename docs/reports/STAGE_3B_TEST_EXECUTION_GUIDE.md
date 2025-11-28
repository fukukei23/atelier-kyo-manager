# Stage 3B テスト実行ガイド

## 問題の原因

現在、PowerShellの問題（`spawn C:\Program Files\PowerShell\7\pwsh.exe ENOENT`）により、直接コマンド実行ができません。

これは、WSL環境でPowerShellが正しく設定されていないか、パスが通っていない可能性があります。

## 解決策

### 方法1: Pythonスクリプトで直接実行（推奨）

PowerShellを経由せず、Pythonスクリプト内で直接pytestを実行します：

```bash
python3 run_tests_direct.py
```

または

```bash
python3 run_stage3b_tests_simple.py
```

### 方法2: WSLターミナルで直接実行

WSLターミナルを開いて、以下のコマンドを実行してください：

```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate
python -m pytest tests/test_telemetry_service_stage3b.py -v --tb=short
```

### 方法3: 環境変数の確認

PowerShellの問題を解決するには、環境変数を確認してください：

```bash
echo $SHELL
which python3
which pytest
```

## テスト準備状況

- ✅ テストファイル: `tests/test_telemetry_service_stage3b.py` (14個のテストケース)
- ✅ 実装完了: TelemetryService クラス、BrowserUseAgent への統合、NavigationDriver への統合
- ✅ 静的解析: エラーなし

## 期待される結果

すべてのテストが成功し、以下が確認できること：

1. ✅ TelemetryService が正しく動作する
2. ✅ NavigationDriver と TelemetryService が統合されている
3. ✅ 既存の observability.py 関数と互換性がある
4. ✅ エラーハンドリングが適切に実装されている
