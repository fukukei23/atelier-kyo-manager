# TODO: MONCLER Drission 診断環境（Windows 実行）再開ポイント

## 状況

- Drission 診断ランナー（診断用スクリプト & HTML/PNG/JSON 保存）はすべて実装済み。

- ただし **WSL では Chrome が動かず、DrissionPage のブラウザ起動に失敗する**ため、

  Windows 環境での実行が必須。

## 次にやるべきこと（保留中）

### 1. プロジェクトを Windows 側にコピーする

- 推奨ディレクトリ例：  

  `C:\Users\USER\tools\atelier-kyo-manager`

### 2. Windows 側で仮想環境を作成する

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
pip install DrissionPage
```

### 3. 診断スクリプトを Windows PowerShell で実行する

推奨スクリプト：

```
scripts/run_moncler_diagnostics_windows.ps1
```

基本コマンド例：

```powershell
python scripts\run_moncler_drission_diagnostics.py --query "down jacket" `
  --target_url "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB" `
  --headless
```

### 4. 診断ログの確認

出力先：

```
artifacts/moncler_drission/<timestamp>/
  - success_plp.html / error_plp.html
  - success_plp.png / error_plp.png
  - success_plp.json / error_plp.json
  - run.log
```

### 5. 次の作業

- 診断ログを元に **Moncler 用 site_config（selectors & navigation）の調整**

- Cloudflare / GeoModal / Cookie の実 DOM 適応改善

- DrissionHandler の微調整

## 備考

- BrowserUseAgent の Playwright ルートには影響なし。

- DrissionPage は "Windows Chrome" 前提なので、WSL 側での実行は不可。

## 完了した実装

- MonclerDrissionHandler（診断モード付き）

- run_moncler_drission_diagnostics.py（診断スクリプト）

- 診断スナップショット（HTML/PNG/JSON）保存機構

- WSL/PowerShell ガイド（docs/WSL_POWERSHELL_EXECUTION_GUIDE.md）

## 再開方法

次に ChatGPT に忘れず再開させるには？

あなたが次に戻ってきたときに ChatGPT にこう言うだけで再開できます：

```
TODO.md の「MONCLER診断環境の保留タスク」から続きを再開したいです。
```

ChatGPT は自動で内容を読み込み、続きを提案・実行できます。

（Cursor も TODO.md を自動で補完対象として扱うので便利）

