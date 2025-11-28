# WSL環境でのシェル設定ガイド

## 問題の原因

CursorがPowerShellを使おうとしていますが、WSL環境ではbashを使うべきです。

エラー: `spawn C:\Program Files\PowerShell\7\pwsh.exe ENOENT`

## 解決策

### 方法1: Cursorの設定を変更（推奨）

`.cursor/settings.json` ファイルを作成し、WSL環境でbashを使うように設定します。

```json
{
  "terminal.integrated.defaultProfile.windows": "WSL",
  "terminal.integrated.profiles.windows": {
    "WSL": {
      "path": "wsl.exe",
      "args": []
    },
    "Bash": {
      "path": "wsl.exe",
      "args": ["-e", "bash"]
    }
  },
  "terminal.integrated.shell.windows": "wsl.exe",
  "terminal.integrated.shellArgs.windows": []
}
```

### 方法2: Pythonスクリプトで直接実行（即座に使える）

PowerShellを経由せず、Pythonスクリプト内で直接pytestを実行します：

```bash
cd /home/yn441611/atelier-kyo-manager
python3 run_tests_direct.py
```

### 方法3: WSLターミナルで直接実行

WSLターミナルを開いて、以下のコマンドを実行してください：

```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate
python -m pytest tests/test_telemetry_service_stage3b.py -v --tb=short
```

## 注意事項

- WSL環境ではPowerShellは必要ありません
- プロジェクトのルールでは「すべての pytest / shell 実行は **WSL Ubuntu** 上を前提とする」と明記されています
- システムレベルの設定変更は不要です

## 確認方法

設定後、以下のコマンドで確認できます：

```bash
echo $SHELL
which python3
which pytest
```

