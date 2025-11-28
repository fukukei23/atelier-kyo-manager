# Chrome/Chromium インストールガイド

## 概要

MONCLER Drission 診断スクリプトを実行するには、Chrome または Chromium が必要です。

## 方法1: Chrome を手動でインストール（推奨）

### Windows でのインストール

1. **Chrome のダウンロード**
   - [Google Chrome 公式サイト](https://www.google.com/chrome/) にアクセス
   - 「Chrome をダウンロード」をクリック
   - インストーラーをダウンロード

2. **インストール**
   - ダウンロードしたインストーラーを実行
   - インストールウィザードに従ってインストール
   - インストール後、Chrome が自動的に起動します

3. **確認**
   - Chrome が正常に起動することを確認
   - バージョンを確認: `chrome://version/`

### インストール先の確認

通常、Chrome は以下の場所にインストールされます：

- **64bit版**: `C:\Program Files\Google\Chrome\Application\chrome.exe`
- **32bit版**: `C:\Program Files (x86)\Google\Chrome\Application\chrome.exe`

## 方法2: Chromium をインストール

### Windows でのインストール

1. **Chromium のダウンロード**
   - [Chromium 公式サイト](https://www.chromium.org/getting-involved/download-chromium) にアクセス
   - または [Chromium ダウンロードページ](https://download-chromium.appspot.com/) を使用

2. **インストール**
   - ダウンロードした ZIP ファイルを解凍
   - 任意の場所に配置（例: `C:\chromium\`）

3. **パスの設定（オプション）**
   - 環境変数 `PATH` に Chromium のパスを追加するか、
   - DrissionPage の設定で Chromium のパスを指定

## 方法3: DrissionPage が自動的にダウンロード（最も簡単）

DrissionPage は、Chrome/Chromium が見つからない場合、自動的に Chromium をダウンロードして使用します。

### 初回実行時の動作

1. **自動ダウンロード**
   - DrissionPage が初回実行時に Chromium を自動ダウンロード
   - ダウンロード先: `~/.DrissionPage/` または `C:\Users\<ユーザー名>\.DrissionPage\`

2. **確認方法**
   ```python
   from DrissionPage import ChromiumPage
   
   # 初回実行時に自動ダウンロードされる
   page = ChromiumPage()
   ```

### 手動で Chromium をダウンロードする場合

DrissionPage のコマンドを使用：

```bash
# DrissionPage の Chromium ダウンロードコマンド
python -c "from DrissionPage import ChromiumPage; ChromiumPage()"
```

または、DrissionPage のツールを使用：

```python
from DrissionPage import ChromiumPage

# 自動的に Chromium をダウンロードして使用
page = ChromiumPage()
```

## 方法4: システムにインストールされている Chrome を使用

既に Chrome がインストールされている場合、DrissionPage は自動的に検出して使用します。

### 確認方法

```python
from DrissionPage import ChromiumPage

# Chrome のパスを確認
page = ChromiumPage()
print(f"Chrome パス: {page.browser.path}")
```

## トラブルシューティング

### Chrome/Chromium が見つからない

1. **パスの確認**
   ```python
   from DrissionPage import ChromiumPage
   
   try:
       page = ChromiumPage()
       print("✅ Chrome/Chromium が見つかりました")
   except Exception as e:
       print(f"❌ エラー: {e}")
   ```

2. **手動でパスを指定**
   ```python
   from DrissionPage import ChromiumPage
   
   # Chrome のパスを明示的に指定
   page = ChromiumPage(browser_path="C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe")
   ```

### ダウンロードが失敗する

1. **ネットワーク接続の確認**
   - インターネット接続を確認
   - ファイアウォールやプロキシの設定を確認

2. **手動でダウンロード**
   - Chrome を手動でインストール（方法1を参照）

### 権限エラー

1. **管理者権限で実行**
   - PowerShell を管理者として実行
   - または、Chrome のインストール先に書き込み権限があることを確認

2. **ユーザーディレクトリの使用**
   - DrissionPage はユーザーディレクトリに Chromium をダウンロードするため、
   - 通常は権限エラーは発生しません

## 推奨される方法

### 最も簡単な方法

**方法3（DrissionPage の自動ダウンロード）** を推奨します。

1. Chrome/Chromium を手動でインストールする必要はありません
2. 初回実行時に自動的にダウンロードされます
3. 設定やパスの指定は不要です

### 実行手順

```bash
# 1. 仮想環境を有効化
.venv\Scripts\activate

# 2. DrissionPage がインストールされていることを確認
python -c "from DrissionPage import ChromiumPage; print('OK')"

# 3. 診断スクリプトを実行（初回実行時に自動的に Chromium をダウンロード）
python scripts\run_moncler_drission_diagnostics.py --query "down jacket" --headless
```

初回実行時、DrissionPage が自動的に Chromium をダウンロードします。数分かかる場合があります。

## 確認方法

### Chrome/Chromium が使用可能か確認

```python
from DrissionPage import ChromiumPage

try:
    page = ChromiumPage()
    print("✅ Chrome/Chromium が使用可能です")
    print(f"ブラウザパス: {page.browser.path}")
    page.quit()
except Exception as e:
    print(f"❌ エラー: {e}")
    print("Chrome/Chromium をインストールするか、DrissionPage に自動ダウンロードさせてください")
```

## まとめ

- **最も簡単**: DrissionPage の自動ダウンロード（方法3）
- **手動インストール**: Chrome を公式サイトからインストール（方法1）
- **既にインストール済み**: 自動的に検出されます（方法4）

通常は、**方法3（自動ダウンロード）** を使用するだけで問題ありません。

