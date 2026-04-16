# WSL環境から直接実行する方法

## 重要な注意事項

⚠️ **WSL環境では実際のブラウザ操作はできません。**

- DrissionPage は Windows 環境でのみ動作します
- WSL環境では、環境確認やコードの検証のみ可能です
- 実際のブラウザ操作を実行するには、Windows環境で実行する必要があります

## WSL環境での実行（環境確認のみ）

WSL環境内では、`wsl` コマンドを使う必要はありません。直接コマンドを実行してください。

### 正しい実行方法

```bash
# WSL環境内で直接実行（環境確認のみ）
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate
python3 scripts/run_moncler_drission_diagnostics.py --query "down jacket" --headless
```

### 間違った実行方法

```bash
# ❌ 間違い: WSL環境内で wsl コマンドを使用
wsl bash -c "cd /home/yn441611/atelier-kyo-manager && ..."
```

WSL環境内では、`wsl` コマンドは不要です。直接 `bash` や `python3` コマンドを使用してください。

## Windows PowerShell から WSL 内のファイルを実行する方法

Windows PowerShell から WSL 内のファイルを実行する場合は、`wsl` コマンドを使用します。

### 方法1: PowerShell スクリプトを使用（推奨）

```powershell
# Windows PowerShell で実行
.\scripts\run_moncler_diagnostics_wsl.ps1 -Query "down jacket" -Headless
```

### 方法2: 直接 wsl コマンドを実行

```powershell
# Windows PowerShell で実行
wsl bash -c "cd /home/yn441611/atelier-kyo-manager && source venv/bin/activate 2>/dev/null || true && python3 scripts/run_moncler_drission_diagnostics.py --query 'down jacket' --headless"
```

## 実行環境の違い

| 環境 | コマンド | 説明 |
|------|---------|------|
| WSL環境内 | `python3 scripts/run_moncler_drission_diagnostics.py` | 直接実行（環境確認のみ） |
| Windows PowerShell | `wsl bash -c "..."` | WSLコマンドを使用 |
| Windows PowerShell | `.\scripts\run_moncler_diagnostics_wsl.ps1` | PowerShellスクリプトを使用 |

## 実際のブラウザ操作を実行するには

実際のブラウザ操作を実行するには、**Windows環境で直接実行**する必要があります：

1. **Windows側にプロジェクトをコピー**
2. **Windows側で仮想環境をセットアップ**
3. **Windows PowerShell で実行**

```powershell
# Windows側のプロジェクトで実行
cd C:\path\to\atelier-kyo-manager
.venv\Scripts\activate
python scripts\run_moncler_drission_diagnostics.py --query "down jacket" --headless
```

## トラブルシューティング

### WSL環境内で wsl コマンドを使った場合

エラー例：
```
wsl: command not found
```

**解決方法**: WSL環境内では `wsl` コマンドは不要です。直接コマンドを実行してください。

```bash
# 正しい（WSL環境内）
python3 scripts/run_moncler_drission_diagnostics.py --query "down jacket" --headless
```

### Windows PowerShell から実行する場合

```powershell
# 正しい（Windows PowerShell）
wsl bash -c "cd /home/yn441611/atelier-kyo-manager && source venv/bin/activate 2>/dev/null || true && python3 scripts/run_moncler_drission_diagnostics.py --query 'down jacket' --headless"
```

## まとめ

1. **WSL環境内**: `wsl` コマンドは不要。直接コマンドを実行
2. **Windows PowerShell**: `wsl` コマンドを使用してWSL内のコマンドを実行
3. **実際のブラウザ操作**: Windows環境で直接実行が必要

