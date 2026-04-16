# Windows PowerShell から WSL内のファイルを実行する方法

## 概要

WSL内のファイル（`\\wsl.localhost\Ubuntu\home\yn441611\atelier-kyo-manager\scripts\run_moncler_drission_diagnostics.py`）を Windows PowerShell から実行する方法です。

## 方法1: WSLコマンドを使用（推奨）

WSL内のPythonと仮想環境を使用する方法です。

### PowerShell スクリプトを使用

```powershell
# WSL経由で実行するスクリプトを使用
.\scripts\run_moncler_diagnostics_wsl.ps1 -Query "down jacket" -Headless
```

### 直接WSLコマンドを実行

```powershell
# Windows PowerShell から WSL内で実行
wsl bash -c "cd /home/yn441611/atelier-kyo-manager && source venv/bin/activate 2>/dev/null || true && python3 scripts/run_moncler_drission_diagnostics.py --query 'down jacket' --headless"
```

**注意**: このコマンドは **Windows PowerShell** で実行してください。WSL環境内で `wsl` コマンドを実行する必要はありません。

## 方法2: WindowsのPythonからWSL内のファイルにアクセス

Windows側のPythonを使用して、WSL内のファイルを実行する方法です。

**注意**: この方法では、WSL内の仮想環境は使用できません。Windows側のPythonとパッケージが必要です。

### PowerShell スクリプトを使用

```powershell
# WSL経由で実行するスクリプトを使用（Windows Python オプション）
.\scripts\run_moncler_diagnostics_wsl.ps1 -Query "down jacket" -Headless
```

スクリプト内で `$useWslPython = $false` に設定すると、Windows側のPythonを使用します。

### 直接実行

```powershell
# WSL内のファイルにアクセス
$scriptPath = "\\wsl.localhost\Ubuntu\home\yn441611\atelier-kyo-manager\scripts\run_moncler_drission_diagnostics.py"

# Windows側のPythonで実行
python $scriptPath --query "down jacket" --headless
```

## 推奨される方法

### 方法1（WSLコマンドを使用）を推奨

理由：
1. WSL内の仮想環境を使用できる
2. WSL内のパッケージ（DrissionPage など）が利用可能
3. パスの問題が少ない

### 実行例

```powershell
# PowerShell スクリプトを使用（最も簡単）
.\scripts\run_moncler_diagnostics_wsl.ps1 -Query "down jacket" -Headless

# ターゲットURLを指定
.\scripts\run_moncler_diagnostics_wsl.ps1 `
  -Query "down jacket" `
  -TargetUrl "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB" `
  -Headless
```

## 重要な注意事項

### WSL環境では実際のブラウザ操作はできません

**WSL環境では、DrissionPage を使用した実際のブラウザ操作はできません。**

- DrissionPage は Windows 環境でのみ動作します
- WSL環境では、環境確認やコードの検証のみ可能です

### 実際のブラウザ操作を実行するには

Windows環境で直接実行する必要があります：

1. **Windows側にプロジェクトをクローン/コピー**
2. **Windows側で仮想環境をセットアップ**
3. **Windows PowerShell で実行**

```powershell
# Windows側のプロジェクトで実行
cd C:\path\to\atelier-kyo-manager
.venv\Scripts\activate
python scripts\run_moncler_drission_diagnostics.py --query "down jacket" --headless
```

## トラブルシューティング

### WSLコマンドが見つからない

```powershell
# WSLがインストールされているか確認
wsl --list --verbose

# WSLがインストールされていない場合、Windows の機能から有効化してください
```

### パスの問題

WSL内のパスとWindowsのパスは異なります：

- **WSL内**: `/home/yn441611/atelier-kyo-manager`
- **Windowsから**: `\\wsl.localhost\Ubuntu\home\yn441611\atelier-kyo-manager`

### 権限の問題

WSL内のファイルにアクセスする場合、適切な権限が必要です。

```powershell
# アクセス権限を確認
Test-Path "\\wsl.localhost\Ubuntu\home\yn441611\atelier-kyo-manager"
```

## まとめ

1. **WSL内で実行**: `wsl` コマンドを使用
2. **Windows側で実行**: Windows側にプロジェクトをコピーして実行（推奨）
3. **実際のブラウザ操作**: Windows環境での実行が必要

