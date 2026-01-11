# Cursor チャット履歴検索ツール

## 概要

Cursorを再起動するたびにチャット履歴が消える問題を調査するためのツールです。
ワークスペースごとに分離されているチャット履歴データベースファイルを検索します。

## 使い方

### 方法1: PowerShellスクリプト（推奨）

WindowsのPowerShellで実行:

```powershell
cd C:\Users\USER\tools\atelier-kyo-manager
.\tools\find_cursor_chat_history.ps1
```

または、プロジェクトルートから:

```powershell
.\tools\find_cursor_chat_history.ps1
```

### 方法2: Pythonスクリプト

WSL環境から実行:

```bash
cd /home/yn441611/atelier-kyo-manager
python3 tools/find_cursor_chat_history.py
```

またはWindowsから直接実行:

```powershell
python tools/find_cursor_chat_history.py
```

## 検索対象

以下のパスにあるプロジェクトごとのチャット履歴を検索します:

- `C:\Users\USER\.cursor\projects\wsl-localhost-Ubuntu-home-yn441611-atelier-kyo-manager`
- `C:\Users\USER\.cursor\projects\c-Users-USER-tools-atelier-kyo-manager`
- `C:\Users\USER\.cursor\projects\wsl-localhost-Ubuntu-home-yn441611-NexusCore`
- その他のプロジェクト

## 出力される情報

1. **プロジェクト一覧**: 見つかったすべてのプロジェクト
2. **データベースファイル**: 各プロジェクト内の `*.db` ファイル
3. **ファイルサイズ**: 大きいファイルがチャット履歴の可能性が高い
4. **更新日時**: 最後に更新された時刻
5. **workspaceStorage ディレクトリ**: ワークスペース固有のストレージ

## 見つかったファイルの確認方法

1. エクスプローラーでファイルのフルパスを開く
2. SQLiteブラウザ（例: DB Browser for SQLite）で開いて内容を確認
3. ファイルサイズが大きいものほど、チャット履歴の可能性が高い

## トラブルシューティング

### スクリプトが実行できない場合

- PowerShellの実行ポリシーを確認:
  ```powershell
  Get-ExecutionPolicy
  ```
- 必要に応じて実行ポリシーを変更:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

### パスが見つからない場合

- `C:\Users\USER\.cursor\projects\` ディレクトリが存在するか確認
- ユーザー名が `USER` でない場合は、スクリプト内のパスを修正

## 関連ファイル

- `CURSOR_CHAT_HISTORY_GUIDE.md`: 詳細な調査方法と対処法

