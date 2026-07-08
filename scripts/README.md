# Scripts ディレクトリ

このディレクトリには、プロジェクトの各種診断・テスト用スクリプトが含まれます。

## Windows 環境での実行

Windows 環境で実行する場合は、以下のスクリプトを使用できます：

- **PowerShell スクリプト（Windows側）**: `run_moncler_diagnostics_windows.ps1`（推奨）
- **PowerShell スクリプト（WSL経由）**: `run_moncler_diagnostics_wsl.ps1`（WSL内のファイルにアクセスする場合）
- **バッチファイル**: `run_moncler_diagnostics_windows.bat`
- **Python スクリプト**: `run_moncler_drission_diagnostics.py`（直接実行）

### 注意事項

⚠️ **WSL環境では実際のブラウザ操作はできません。**

- DrissionPage は Windows 環境でのみ動作します
- WSL環境では、環境確認やコードの検証のみ可能です
- 実際のブラウザ操作を実行するには、Windows環境で直接実行してください

詳細は以下を参照してください：
- **Windows環境での実行**: `docs/WINDOWS_EXECUTION_GUIDE.md`
- **WSL経由での実行**: `docs/WSL_POWERSHELL_EXECUTION_GUIDE.md`

## スクリプト一覧

### `run_moncler_drission_diagnostics.py`

MONCLER 専用の DrissionPage ルートを単体で実行し、診断情報を保存するスクリプトです。

#### 機能

- MonclerDrissionHandler を単体で実行
- 成功・失敗それぞれについて以下を保存:
  - HTML スナップショット
  - スクリーンショット (PNG)
  - 抽出結果 or エラー情報の JSON
  - ログファイル (run.log)

#### 使用方法

```bash
# 基本的な実行
python scripts/run_moncler_drission_diagnostics.py \
  --query "down jacket" \
  --target_url "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB" \
  --headless

# 複数回実行
python scripts/run_moncler_drission_diagnostics.py \
  --query "jacket" \
  --runs 3 \
  --out_base "artifacts/moncler_test"

# ヘッドフルモード（ブラウザを表示）
python scripts/run_moncler_drission_diagnostics.py \
  --query "down jacket" \
  --headful
```

#### 引数

- `--query`: 検索クエリ（デフォルト: "down jacket"）
- `--target_url`: 直接指定する PLP URL（オプション）
- `--headless`: ヘッドレスモードで実行
- `--runs`: 実行回数（デフォルト: 1）
- `--out_base`: 出力ベースディレクトリ（デフォルト: "artifacts/moncler_drission"）

#### 出力先

各実行ごとに `artifacts/moncler_drission/YYYYMMDD_HHMMSS/` ディレクトリが作成され、以下が保存されます:

- `success_plp.html` / `.png` / `.json` (成功時)
- `error_plp.html` / `.png` / `.json` (失敗時)
- `error_no_items.html` / `.png` / `.json` (商品が見つからない場合)
- `run.log` (ログファイル)

#### 注意事項

- DrissionPage がインストールされている必要があります
- Windows 環境での実行を推奨します（DrissionPage は Windows 環境で動作します）
- 診断モードは自動的に有効化されます（`debug=True`）
- **Chrome/Chromium**: 手動でインストールする必要はありません。DrissionPage が初回実行時に自動的にダウンロードします

