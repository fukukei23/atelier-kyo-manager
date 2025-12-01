# 「WSL環境に直接ログイン」とは？

## 簡単に言うと

「WSL環境に直接ログイン」とは、**Windows から WSL（Linux環境）に入って、そこで直接コマンドを実行すること**です。

## イメージ図

### 現在（Cursor AI ツール経由）

```
あなた
  └─ Cursor AI のチャットで「テストして」と入力
      └─ Cursor AI が run_terminal_cmd ツールを使って実行
          └─ 出力が表示されない ❌
```

### 「直接ログイン」した場合

```
あなた
  └─ あなた自身が WSL 環境に入る
      └─ あなたが直接コマンドを入力・実行
          └─ 出力が正常に表示される ✅
```

## 具体的な手順

### 最も簡単な方法: Cursor のターミナルを使う

Cursor には統合ターミナルがあります。`.cursor/settings.json` でデフォルトが WSL に設定されているため、ターミナルを開けば自動的に WSL 環境になります。

1. **Cursor のターミナルを開く**
   - `Ctrl + `` （バッククォート、キーボードの左上のキー）を押す
   - または、メニューから「Terminal」→「New Terminal」を選択

2. **WSL環境に入る**
   - 自動的に WSL 環境（Ubuntu）に入ります
   - プロンプトが `yn441611@DESKTOP-XXXXX:~/atelier-kyo-manager$` のような形式になります

3. **コマンドを実行**
   ```bash
   cd /home/yn441611/atelier-kyo-manager
   source venv/bin/activate
   python -m pytest tests/test_product_extractor.py -v
   ```

### 他の方法: Windows PowerShell から

1. **Windows PowerShell を開く**
   - Windows キーを押して「PowerShell」と検索
   - 「Windows PowerShell」を開く

2. **`wsl` コマンドを実行**
   ```powershell
   wsl
   ```

3. **プロンプトが変わることを確認**
   ```
   # 実行前（Windows PowerShell）
   PS C:\Users\YourName> 
   
   # 実行後（WSL環境に入った）
   yn441611@DESKTOP-XXXXX:/mnt/c/Users/YourName$ 
   ```

4. **プロジェクトディレクトリに移動してコマンドを実行**
   ```bash
   cd /home/yn441611/atelier-kyo-manager
   source venv/bin/activate
   python -m pytest tests/test_product_extractor.py -v
   ```

## 違いの比較

### 「ツール経由」の場合

```
あなた → Cursor AI のチャット → 「テストして」と入力
         ↓
     Cursor AI が run_terminal_cmd ツールを使って実行
         ↓
     コマンドは実行されるが、出力が表示されない ❌
```

**特徴**:
- Cursor のチャットから「テストして」と言うだけで実行できる
- ただし、出力が表示されない

### 「直接ログイン」の場合

```
あなた → Cursor のターミナル（または PowerShell）で WSL に入る
         ↓
     あなたが直接コマンドを入力・実行
         ↓
     出力が正常に表示される ✅
```

**特徴**:
- あなた自身が WSL 環境に入ってコマンドを実行する
- 出力が正常に表示される
- インタラクティブにコマンドを実行できる

## 確認方法

WSL環境に入れたかどうかを確認するには：

```bash
# OSを確認
uname -a
# 出力例: Linux DESKTOP-XXXXX 5.10.16.3-microsoft-standard-WSL2 ...
# → "Linux" と表示されれば WSL環境です ✅

# 現在のディレクトリを確認
pwd
# 出力例: /home/yn441611/atelier-kyo-manager
# → "/home/..." のようなパスなら WSL環境です ✅
```

## まとめ

「WSL環境に直接ログイン」とは：

1. **Cursor のターミナルを開く**（`Ctrl + ``）または **Windows PowerShell で `wsl` コマンドを実行**
2. **WSL環境に入る**（プロンプトが変わる）
3. **あなたが直接コマンドを入力・実行する**
4. **出力が正常に表示される**

これにより、Cursor AI のツールの制約を回避し、テスト結果を直接確認できます。

## 関連ドキュメント

- `docs/reports/WSL_ACCESS_GUIDE.md` - WSL環境へのアクセス方法の詳細
