# Cursor チャット履歴検索結果

## 実行方法

ターミナルが正常に動作しないため、以下のいずれかの方法で実行してください：

### 方法1: Pythonスクリプトを直接実行

WSLターミナルまたはWindowsのコマンドプロンプトで：

```bash
cd /home/yn441611/atelier-kyo-manager
python3 execute_find_chat_history.py
```

またはWindowsから：

```powershell
cd C:\Users\USER\tools\atelier-kyo-manager
python execute_find_chat_history.py
```

### 方法2: PowerShellスクリプト

Windows PowerShellで：

```powershell
cd C:\Users\USER\tools\atelier-kyo-manager
.\tools\find_cursor_chat_history.ps1
```

## 検索対象

- プロジェクト: `wsl-localhost-Ubuntu-home-yn441611-atelier-kyo-manager`
- プロジェクト: `c-Users-USER-tools-atelier-kyo-manager`
- プロジェクト: `wsl-localhost-Ubuntu-home-yn441611-NexusCore`

## 検索条件

- 更新日時が昨日以降のデータベースファイル（*.db）を優先表示
- 今朝以降に更新されたファイルは ⭐ マークで強調表示

## 結果

スクリプトを実行すると、以下の情報が表示されます：

1. 見つかったプロジェクト一覧
2. 各プロジェクト内のデータベースファイル
3. 更新日時順にソートされたファイルリスト
4. 最近更新されたファイル（再起動後のチャット履歴の可能性が高い）

---

**注意**: ターミナルが正常に動作しない場合は、手動でエクスプローラーから確認してください：
- `C:\Users\USER\.cursor\projects\wsl-localhost-Ubuntu-home-yn441611-atelier-kyo-manager`
- `C:\Users\USER\.cursor\projects\c-Users-USER-tools-atelier-kyo-manager`

