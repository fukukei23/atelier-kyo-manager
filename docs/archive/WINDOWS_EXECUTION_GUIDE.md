# Windows 環境での MONCLER Drission 診断スクリプト実行ガイド

## 前提条件

1. **Python のインストール**
   - Python 3.8 以上がインストールされていること
   - パスが通っていること

2. **DrissionPage のインストール**
   ```bash
   pip install DrissionPage
   ```

3. **Chrome/Chromium のインストール**
   - Chrome または Chromium がインストールされていること
   - **注意**: DrissionPage は初回実行時に自動的に Chromium をダウンロードします
   - 手動インストールは不要です（詳細は `docs/CHROME_CHROMIUM_INSTALLATION_GUIDE.md` を参照）

4. **仮想環境の準備（推奨）**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   pip install DrissionPage
   ```

## 実行前の確認

実行前に、環境が整っているか確認することを推奨します：

```bash
# 環境確認スクリプトを実行
python scripts\check_windows_environment.py
```

すべての確認が完了したら、以下の方法で実行できます。

## 実行方法

### 方法1: 簡単なバッチファイルを使用（最も簡単）

```cmd
# 環境確認と実行を自動で行う
scripts\run_diagnostics_simple.bat
```

### 方法2: PowerShell スクリプトを使用（推奨）

```powershell
# 基本的な実行
.\scripts\run_moncler_diagnostics_windows.ps1 -Query "down jacket" -Headless

# ターゲットURLを指定
.\scripts\run_moncler_diagnostics_windows.ps1 `
  -Query "down jacket" `
  -TargetUrl "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB" `
  -Headless

# 複数回実行
.\scripts\run_moncler_diagnostics_windows.ps1 `
  -Query "jacket" `
  -Runs 3 `
  -OutBase "artifacts\moncler_test"
```

### 方法2: バッチファイルを使用

```cmd
REM 基本的な実行
scripts\run_moncler_diagnostics_windows.bat --query "down jacket" --headless

REM ターゲットURLを指定
scripts\run_moncler_diagnostics_windows.bat ^
  --query "down jacket" ^
  --target_url "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB" ^
  --headless

REM 複数回実行
scripts\run_moncler_diagnostics_windows.bat ^
  --query "jacket" ^
  --runs 3 ^
  --out_base "artifacts\moncler_test"
```

### 方法3: Python スクリプトを直接実行

```bash
# 仮想環境を有効化（推奨）
.venv\Scripts\activate

# 基本的な実行
python scripts\run_moncler_drission_diagnostics.py --query "down jacket" --headless

# ターゲットURLを指定
python scripts\run_moncler_drission_diagnostics.py ^
  --query "down jacket" ^
  --target_url "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB" ^
  --headless

# 複数回実行
python scripts\run_moncler_drission_diagnostics.py ^
  --query "jacket" ^
  --runs 3 ^
  --out_base "artifacts\moncler_test"
```

## 出力先

各実行ごとに `artifacts\moncler_drission\YYYYMMDD_HHMMSS\` ディレクトリが作成され、以下が保存されます：

- `success_plp.html` / `.png` / `.json` (成功時)
- `error_plp.html` / `.png` / `.json` (失敗時)
- `error_no_items.html` / `.png` / `.json` (商品が見つからない場合)
- `run.log` (ログファイル)

## トラブルシューティング

### PowerShell スクリプトの実行ポリシーエラー

```powershell
# 実行ポリシーを確認
Get-ExecutionPolicy

# 実行ポリシーを変更（管理者権限が必要）
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### DrissionPage のインポートエラー

```bash
# DrissionPage がインストールされているか確認
python -c "from DrissionPage import ChromiumPage; print('OK')"

# インストールされていない場合
pip install DrissionPage
```

### Chrome/Chromium が見つからない

- **自動ダウンロード**: DrissionPage は初回実行時に自動的に Chromium をダウンロードします
- **手動インストール**: Chrome を公式サイトからインストールすることも可能です
- **詳細**: `docs/CHROME_CHROMIUM_INSTALLATION_GUIDE.md` を参照してください

### 仮想環境の問題

```bash
# 仮想環境を再作成
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install DrissionPage
```

## 実行例

### 例1: 基本的な実行（ヘッドレスモード）

```powershell
.\scripts\run_moncler_diagnostics_windows.ps1 -Query "down jacket" -Headless
```

### 例2: ブラウザを表示して実行

```powershell
.\scripts\run_moncler_diagnostics_windows.ps1 -Query "down jacket"
```

### 例3: ターゲットURLを指定

```powershell
.\scripts\run_moncler_diagnostics_windows.ps1 `
  -Query "down jacket" `
  -TargetUrl "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB" `
  -Headless
```

### 例4: 複数回実行して結果を比較

```powershell
.\scripts\run_moncler_diagnostics_windows.ps1 `
  -Query "jacket" `
  -Runs 3 `
  -OutBase "artifacts\moncler_test"
```

## 結果の確認

実行後、`artifacts\moncler_drission\` ディレクトリ内にタイムスタンプ付きのディレクトリが作成されます。

各ディレクトリには以下が含まれます：

1. **HTML ファイル**: ページの HTML ソース
2. **PNG ファイル**: スクリーンショット
3. **JSON ファイル**: 抽出結果またはエラー情報
4. **ログファイル**: 実行ログ

これらのファイルを確認することで、以下を分析できます：

- どのセレクタが効いているか
- どこで処理が詰まっているか
- エラーの原因

## 注意事項

1. **初回実行時**: DrissionPage が自動的に Chromium をダウンロードします（数分かかる場合があります）
2. **ネットワーク**: インターネット接続が必要です（初回の Chromium ダウンロード時）
3. **実行時間**: 1回の実行に数分かかる場合があります
4. **ディスク容量**: 
   - Chromium のダウンロード: 約 100-200MB
   - スクリーンショットとHTMLファイル: 実行ごとに数MB
   - 十分な容量を確保してください

## Chrome/Chromium について

**重要**: 手動で Chrome/Chromium をインストールする必要はありません。

DrissionPage は初回実行時に自動的に Chromium をダウンロードして使用します。

詳細は `docs/CHROME_CHROMIUM_INSTALLATION_GUIDE.md` を参照してください。

