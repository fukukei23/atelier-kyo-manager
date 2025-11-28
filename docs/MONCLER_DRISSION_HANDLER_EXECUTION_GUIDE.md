# MonclerDrissionHandler 実行ガイド

## 概要

`MonclerDrissionHandler` は Windows 環境で実行する必要があります。WSL 環境では実行できない可能性があります。

## 実行方法

### 1. DrissionPage のインストール確認

```bash
pip install DrissionPage
```

### 2. Windows環境での実行

```bash
# Windows PowerShell またはコマンドプロンプトで実行
python tools/run_moncler_drission_test.py
```

### 3. テストスクリプトの説明

#### `tools/test_moncler_drission_handler.py`

基本的なテストスクリプト。対話的な実行向け。

#### `tools/run_moncler_drission_test.py`

ログファイルに結果を保存する形式。WSL環境でも実行可能（ただし、実際のブラウザ操作はWindows環境が必要）。

## 実行結果の確認

### ログファイル

実行後、以下のファイルが生成されます：

- `moncler_drission_test_YYYYMMDD_HHMMSS.log`: 詳細なログ

### 結果ファイルの確認

```bash
# 最新のログファイルを確認
ls -lt moncler_drission_test_*.log | head -1

# ログファイルの内容を確認
cat moncler_drission_test_*.log | tail -100
```

## 注意事項

### Windows環境必須

- DrissionPage はローカルの Chrome / Chromium を使用するため、Windows環境で実行する必要があります
- WSL環境からWindows側のChromeを呼び出すことは可能ですが、設定が必要です

### 実行環境の確認

テストスクリプトは自動的に以下をチェックします：

1. DrissionPage のインストール確認
2. RunContext の作成
3. サイト設定の読み込み
4. MonclerDrissionHandler の初期化

### エラーハンドリング

- DrissionPage がインストールされていない場合: インストール方法を表示
- 設定ファイルが見つからない場合: デフォルト設定を使用
- ブラウザ起動エラー: エラーメッセージとトレースバックを表示

## トラブルシューティング

### DrissionPage がインストールされていない

```
❌ DrissionPage がインストールされていません
   以下のコマンドでインストールしてください:
   pip install DrissionPage
```

**解決方法:**
```bash
pip install DrissionPage
```

### Chrome がインストールされていない

DrissionPage は Chrome または Chromium が必要です。Windows環境にChromeがインストールされていることを確認してください。

### 設定ファイルが見つからない

デフォルト設定が使用されますが、`app/config/sites/overrides.local.json` に `MONCLER_OFFICIAL` の設定があることを確認してください。

## 実行フロー

1. **DrissionPage の確認**
   - インポート可能かチェック
   - 不可能な場合はエラーメッセージを表示

2. **RunContext の作成**
   - 実行コンテキストを初期化
   - ログやファイル保存先を設定

3. **サイト設定の読み込み**
   - `MONCLER_OFFICIAL` の設定を読み込み
   - 見つからない場合はデフォルト設定を使用

4. **MonclerDrissionHandler の作成**
   - ハンドラを初期化
   - user_data_path を設定

5. **テスト実行**
   - クエリを指定して実行
   - 商品情報を取得

6. **結果の表示**
   - 成功/失敗の表示
   - 取得した商品情報の表示

## 結果の解釈

### 成功した場合

```
✅ テスト成功
取得した商品数: 5
```

### 失敗した場合

```
❌ テスト失敗
メッセージ: 商品が見つかりませんでした
```

詳細はログファイルを確認してください。

## 参考リンク

- [DrissionPage統合ドキュメント](docs/DRISSIONPAGE_INTEGRATION.md)
- [MonclerDrissionHandler実装](app/specialized/moncler_handler.py)

