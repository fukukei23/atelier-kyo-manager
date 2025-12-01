# WSL環境へのアクセス方法

## 「WSL環境に直接ログイン」とは？

「WSL環境に直接ログイン」とは、**Windows から WSL（Linux）環境に入って、そこでコマンドを実行する**ことを意味します。

## WSL環境に入る方法

### 方法1: Windows PowerShell または CMD から

1. **Windows PowerShell を開く**（または CMD）
   - Windows キー + X → 「Windows PowerShell」を選択
   - または、スタートメニューで「PowerShell」を検索

2. **`wsl` コマンドを実行**
   ```powershell
   wsl
   ```

3. **WSL環境に入る**
   - プロンプトが `username@hostname:/mnt/c/Users/...$` のような形式に変わります
   - これが WSL 環境（Ubuntu）に入った状態です

4. **プロジェクトディレクトリに移動**
   ```bash
   cd /home/yn441611/atelier-kyo-manager
   ```

5. **仮想環境を有効化**
   ```bash
   source venv/bin/activate
   ```

6. **コマンドを実行**
   ```bash
   python -m pytest tests/test_product_extractor.py -v
   ```

### 方法2: Windows Terminal から

1. **Windows Terminal を開く**
   - Windows キー + X → 「Windows Terminal」を選択
   - または、スタートメニューで「Terminal」を検索

2. **WSL タブを開く**
   - タブの横の「+」ボタンをクリック → 「Ubuntu」を選択
   - または、`Ctrl + Shift + T` で新しいタブを開いて「Ubuntu」を選択

3. **WSL環境に入る**
   - 自動的に WSL 環境に入ります

4. **以降は方法1の手順4以降と同じ**

### 方法3: Cursor の統合ターミナルから

1. **Cursor のターミナルを開く**
   - `Ctrl + `` （バッククォート）でターミナルを開く
   - または、メニューから「Terminal」→「New Terminal」を選択

2. **WSL を選択**
   - ターミナルの右上の「+」ボタンの横にあるドロップダウンから「Ubuntu」を選択
   - または、`.cursor/settings.json` で設定されているデフォルトプロファイル（WSL）が使用されます

3. **WSL環境に入る**
   - 自動的に WSL 環境に入ります

4. **以降は方法1の手順4以降と同じ**

## 実際の例

### Windows PowerShell から実行する場合

```powershell
# Windows PowerShell で
PS C:\Users\YourName> wsl

# WSL環境に入る（プロンプトが変わる）
yn441611@DESKTOP-XXXXX:/mnt/c/Users/YourName$ cd /home/yn441611/atelier-kyo-manager
yn441611@DESKTOP-XXXXX:~/atelier-kyo-manager$ source venv/bin/activate
(venv) yn441611@DESKTOP-XXXXX:~/atelier-kyo-manager$ python -m pytest tests/test_product_extractor.py -v
```

### WSL環境内で直接実行する場合

```bash
# 既に WSL環境にいる場合
yn441611@DESKTOP-XXXXX:~/atelier-kyo-manager$ python -m pytest tests/test_product_extractor.py -v
```

## 「直接ログイン」と「ツール経由」の違い

### ツール経由（Cursor AI の `run_terminal_cmd`）

```
Windows → Cursor AI → run_terminal_cmd ツール → WSL → コマンド実行
                                                         ↓
                                                   出力が表示されない
```

- コマンドは正常に実行される
- ただし、出力が表示されない（ツールの制約）

### 直接ログイン

```
Windows → WSL環境にログイン → コマンド実行
                              ↓
                           出力が正常に表示される
```

- コマンドを直接実行できる
- 出力が正常に表示される
- 最も確実な方法

## 確認方法

WSL環境に入れたかどうかを確認するには：

```bash
# プロンプトを確認
# WSL環境の場合: username@hostname:~/atelier-kyo-manager$
# Windows PowerShell の場合: PS C:\Users\YourName>

# OSを確認
uname -a
# Linux DESKTOP-XXXXX ... と表示されれば WSL環境

# 現在のディレクトリを確認
pwd
# /home/yn441611/atelier-kyo-manager と表示されれば WSL環境
```

## まとめ

「WSL環境に直接ログイン」とは：

1. **Windows から WSL 環境に入る**（`wsl` コマンドまたは Windows Terminal を使用）
2. **WSL環境内でコマンドを直接実行する**
3. **出力が正常に表示される**

これにより、Cursor AI のツールの制約を回避し、テスト結果を直接確認できます。

