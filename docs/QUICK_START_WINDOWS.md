# Windows 環境でのクイックスタートガイド

## ⚠️ 重要な注意事項

**WSL環境では実際のブラウザ操作はできません。**

- DrissionPage は Windows 環境でのみ動作します
- WSL環境では、環境確認やコードの検証のみ可能です
- 実際のブラウザ操作を実行するには、**Windows PowerShell またはコマンドプロンプト**を使用してください

詳細は `docs/WSL_WINDOWS_EXECUTION_NOTE.md` を参照してください。

## 最も簡単な実行方法

### ステップ1: 環境確認

```cmd
python scripts\check_windows_environment.py
```

すべての確認が完了したら、次のステップに進みます。

### ステップ2: 実行

#### 方法A: 簡単なバッチファイル（推奨）

```cmd
scripts\run_diagnostics_simple.bat
```

このバッチファイルは、環境確認と実行を自動で行います。

#### 方法B: PowerShell スクリプト

```powershell
.\scripts\run_moncler_diagnostics_windows.ps1 -Query "down jacket" -Headless
```

#### 方法C: Python スクリプトを直接実行

```bash
# 仮想環境を有効化（推奨）
.venv\Scripts\activate

# 実行
python scripts\run_moncler_drission_diagnostics.py --query "down jacket" --headless
```

## 初回実行時の注意

1. **Chromium の自動ダウンロード**
   - 初回実行時に DrissionPage が自動的に Chromium をダウンロードします
   - 数分かかる場合があります
   - インターネット接続が必要です

2. **実行時間**
   - 1回の実行に数分かかる場合があります
   - ブラウザの起動、ページの読み込み、商品情報の抽出に時間がかかります

## 出力先

実行後、以下のディレクトリに結果が保存されます：

```
artifacts\moncler_drission\YYYYMMDD_HHMMSS\
  - success_plp.html / .png / .json (成功時)
  - error_plp.html / .png / .json (失敗時)
  - run.log (ログファイル)
```

## トラブルシューティング

### 環境確認でエラーが出た場合

1. **DrissionPage が未インストール**
   ```bash
   pip install DrissionPage
   ```

2. **Python バージョンが古い**
   - Python 3.8 以上が必要です
   - `python --version` で確認

3. **サイト設定が見つからない**
   - `app/config/sites/overrides.local.json` に MONCLER_OFFICIAL の設定があるか確認

### 実行時にエラーが出た場合

1. **Chromium のダウンロードエラー**
   - インターネット接続を確認
   - ファイアウォールやプロキシの設定を確認

2. **権限エラー**
   - PowerShell を管理者として実行
   - または、ユーザーディレクトリに書き込み権限があることを確認

## 詳細情報

- **実行ガイド**: `docs/WINDOWS_EXECUTION_GUIDE.md`
- **Chrome/Chromium インストール**: `docs/CHROME_CHROMIUM_INSTALLATION_GUIDE.md`
- **スクリプトの説明**: `scripts/README.md`

