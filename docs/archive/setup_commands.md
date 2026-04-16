# Atelier Kyo Manager - セットアップコマンド

## 環境構築

### Windows (PowerShell)

```powershell
# 1. 仮想環境作成
python -m venv .venv

# 2. 仮想環境有効化
.\.venv\Scripts\activate

# 3. ライブラリインストール
pip install -r requirements.txt

# 4. Webアプリ起動
flask run
```

### macOS / Linux

```bash
# 1. 仮想環境作成
python -m venv .venv

# 2. 仮想環境有効化
source .venv/bin/activate

# 3. ライブラリインストール
pip install -r requirements.txt

# 4. Webアプリ起動
flask run
```

## 重要な手動作業

`.env`ファイルの作成とAPIキーの設定を完了後、`flask run`でサーバーを起動してください。

## テスト実行

```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate
python -m pytest tests/
```
