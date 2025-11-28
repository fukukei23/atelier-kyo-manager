# WSL環境とWindows環境での実行について

## 重要な注意事項

### WSL環境では実際のブラウザ操作はできません

WSL環境では、DrissionPage を使用した実際のブラウザ操作は**できません**。

- DrissionPage は Windows 環境でのみ動作します
- WSL環境では、環境確認やコードの検証のみ可能です

## WSL環境での確認

WSL環境では、以下の確認のみ可能です：

```bash
# 環境確認（WSL環境）
python3 scripts/check_windows_environment.py
```

この確認では、以下をチェックします：
- Python バージョン
- DrissionPage のインストール状況
- サイト設定の読み込み
- RunContext の初期化
- MonclerDrissionHandler の初期化

**ただし、実際のブラウザ操作（Chrome/Chromium の起動）は Windows 環境で実行する必要があります。**

## Windows環境での実行

実際のブラウザ操作を実行するには、**Windows PowerShell またはコマンドプロンプト**を使用してください。

### Windows環境での実行方法

#### 方法1: 簡単なバッチファイル（推奨）

Windows PowerShell またはコマンドプロンプトで：

```cmd
scripts\run_diagnostics_simple.bat
```

#### 方法2: PowerShell スクリプト

```powershell
.\scripts\run_moncler_diagnostics_windows.ps1 -Query "down jacket" -Headless
```

#### 方法3: Python スクリプトを直接実行

```bash
# 仮想環境を有効化
.venv\Scripts\activate

# 実行（Windows環境ではスラッシュではなくバックスラッシュを使用）
python scripts\run_moncler_drission_diagnostics.py --query "down jacket" --headless
```

## パスの違い

### WSL環境（Linux形式）
```bash
python3 scripts/run_moncler_drission_diagnostics.py --query "down jacket" --headless
```

### Windows環境（Windows形式）
```cmd
python scripts\run_moncler_drission_diagnostics.py --query "down jacket" --headless
```

**注意**: WSL環境ではバックスラッシュ（\）は使えません。スラッシュ（/）を使用してください。

## 実行環境の確認

現在の環境を確認するには：

```bash
# WSL環境かどうかを確認
uname -a

# Windows環境の場合、PowerShellで確認
$PSVersionTable
```

## 推奨されるワークフロー

1. **WSL環境**: コードの編集、環境確認、テスト
2. **Windows環境**: 実際のブラウザ操作の実行

WSL環境でコードを編集し、Windows環境で実際のブラウザ操作を実行することを推奨します。

## トラブルシューティング

### WSL環境でバックスラッシュを使った場合

エラー例：
```
python: can't open file '/home/yn441611/atelier-kyo-manager/scriptsrun_moncler_drission_diagnostics.py': [Errno 2] No such file or directory
```

**解決方法**: スラッシュ（/）を使用してください。

```bash
# 正しい（WSL環境）
python3 scripts/run_moncler_drission_diagnostics.py --query "down jacket" --headless

# 間違い（WSL環境でバックスラッシュを使用）
python3 scripts\run_moncler_drission_diagnostics.py --query "down jacket" --headless
```

### Windows環境でスラッシュを使った場合

Windows環境では、スラッシュ（/）もバックスラッシュ（\）も使用できますが、バックスラッシュ（\）が推奨されます。

```cmd
# どちらも動作しますが、バックスラッシュが推奨
python scripts\run_moncler_drission_diagnostics.py --query "down jacket" --headless
python scripts/run_moncler_drission_diagnostics.py --query "down jacket" --headless
```

