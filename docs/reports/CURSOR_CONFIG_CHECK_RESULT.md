# Cursor設定ファイル確認結果

## 実行方法

以下のいずれかの方法でスクリプトを実行してください：

### 方法1: 直接実行（推奨）

```bash
cd /home/yn441611/atelier-kyo-manager
python3 tools/check_cursor_config.py
```

または Windows PowerShell で：

```powershell
cd C:\Users\USER\tools\atelier-kyo-manager
python tools\check_cursor_config.py
```

### 方法2: 実行ラッパーを使用

```bash
python3 execute_check_cursor_config.py
```

## 確認内容

このスクリプトは以下の内容を確認します：

1. **extensions ディレクトリ** - インストールされている拡張機能の一覧
2. **projects ディレクトリ** - すべてのプロジェクトと、各プロジェクト内の .db ファイル
3. **argv.json ファイル** - Cursorの起動引数や設定
4. **ide_state.json ファイル** - IDEの状態情報
5. **チャット履歴関連ファイル** - *.db、*chat*.db、*history*.db などの検索

## 期待される結果

- 各ディレクトリの内容
- 各ファイルの内容（JSON形式で表示）
- 最近更新されたデータベースファイルの一覧

## 注意事項

- 読み取り専用の操作のみを行います
- ファイルを変更することはありません
- エラーが発生した場合は、エラーメッセージが表示されます

