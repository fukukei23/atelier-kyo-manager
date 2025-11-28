# Cursor チャット履歴の保存場所と確認方法

## 問題
Cursorを再起動するたびに、チャット履歴が消えてしまう。

## チャット履歴の保存場所

Cursorのチャット履歴は、プロジェクトごとに以下の場所に保存されています：

```
C:\Users\USER\.cursor\projects\<プロジェクト名>\
```

### 現在のプロジェクトの保存場所

1. **atelier-kyo-manager (WSL経由)**
   ```
   C:\Users\USER\.cursor\projects\wsl-localhost-Ubuntu-home-yn441611-atelier-kyo-manager
   ```

2. **atelier-kyo-manager (Windows直接パス)**
   ```
   C:\Users\USER\.cursor\projects\c-Users-USER-tools-atelier-kyo-manager
   ```

3. **NexusCore**
   ```
   C:\Users\USER\.cursor\projects\wsl-localhost-Ubuntu-home-yn441611-NexusCore
   ```

## 確認手順

### 1. エクスプローラーで確認

1. Windowsのエクスプローラーを開く
2. アドレスバーに以下を入力してEnter:
   ```
   C:\Users\USER\.cursor\projects\wsl-localhost-Ubuntu-home-yn441611-atelier-kyo-manager
   ```
3. 以下のファイル/ディレクトリを探す:
   - `*.db` ファイル（SQLiteデータベース）
   - `workspaceStorage/` ディレクトリ
   - `globalStorage/` ディレクトリ

### 2. チャット履歴データベースの場所

通常、以下のいずれかに保存されています：

- `workspaceStorage/*/state.vscdb` - ワークスペースの状態
- `globalStorage/*/chat-history.db` - チャット履歴
- `*.cursor-chat-history.db` - チャット履歴専用ファイル

## 履歴が消える原因と対処法

### 原因1: プロジェクトが正しく開かれていない

**問題**: 異なるパスでプロジェクトを開いている

**対処法**:
- WSL経由で開く場合: `\\wsl.localhost\Ubuntu\home\yn441611\atelier-kyo-manager`
- Windows直接パスで開く場合: `C:\Users\USER\tools\atelier-kyo-manager`
- どちらか一方に統一する

### 原因2: データベースファイルが削除されている

**確認方法**:
- 上記のパスで `.db` ファイルが存在するか確認
- ファイルサイズが0バイトでないか確認

**対処法**:
- バックアップがあれば復元
- なければ履歴は復旧不可（今後の予防策を適用）

### 原因3: Cursorの設定で履歴がクリアされている

**確認方法**:
1. Cursorで `Ctrl+,` を押して設定を開く
2. 検索ボックスで "chat" または "history" を検索
3. 以下の設定を確認:
   - `cursor.chat.history.enabled`
   - `cursor.chat.history.retentionDays`

**対処法**:
- 履歴保持期間を延長
- 履歴を無効化していないか確認

### 原因4: ワークスペースごとに履歴が分離されている

**問題**: ワークスペースを切り替えると、履歴も切り替わる

**対処法**:
- 同じプロジェクトでも、開き方が異なると別の履歴になる
- 常に同じ方法でプロジェクトを開く

## 予防策

### 1. チャット履歴の手動バックアップ

重要なチャットは定期的にエクスポート：

1. Cursorのチャット画面で履歴アイコンをクリック
2. または `Alt+Ctrl+'` を押す
3. エクスポート機能（あれば）を使用
4. または、重要な内容を手動でMarkdownファイルにコピー

### 2. プロジェクトの開き方を統一

- WSL経由の場合は常にWSL経由で開く
- Windows直接パスの場合は常にWindows直接パスで開く

### 3. データベースファイルのバックアップ

定期的に以下をバックアップ：
```
C:\Users\USER\.cursor\projects\wsl-localhost-Ubuntu-home-yn441611-atelier-kyo-manager\**\*.db
```

## トラブルシューティング

### チャット履歴が表示されない場合

1. Cursorを完全に終了
2. データベースファイルのパーミッションを確認
3. Cursorを再起動
4. プロジェクトを再オープン

### データベースファイルが破損している場合

1. バックアップから復元
2. なければ履歴は復旧不可
3. 今後の予防策を適用

## 参考情報

- Cursorの公式ドキュメント: https://docs.cursor.com
- チャット履歴機能: https://docs.cursor.com/ja/agent/chat/history

