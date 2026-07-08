# WSL環境でのコマンド出力取得の改善案

## 現状の問題

WSL環境で `run_terminal_cmd` ツールを使用しても、コマンド出力が取得できていません。これは以下の原因が考えられます：

1. **WSLとCursorの統合制限**: WSL環境でのコマンド実行時に、標準出力が正しくキャプチャされない
2. **バッファリング**: Pythonの出力バッファリングにより、出力が即座に表示されない
3. **シェルの設定**: シェルの設定により、出力がリダイレクトされている

## 改善案

### 案1: ファイルベースの出力取得（推奨）

コマンド実行時に、出力をファイルに書き込んでから読み取る方法：

```python
# コマンド実行時に必ずファイルに出力
python test_site_config_connection.py > test_output.log 2>&1

# その後、ファイルを読み取る
cat test_output.log
```

### 案2: Pythonスクリプト内でログファイルを生成

テストスクリプト自体がログファイルを生成するようにする：

```python
import logging

# ファイルハンドラーを追加
file_handler = logging.FileHandler('test_results.log')
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
```

### 案3: 環境変数の設定

バッファリングを無効化：

```bash
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
```

### 案4: 直接ターミナルで実行

Cursorのツールではなく、WSLターミナルで直接実行する方法が最も確実です。

## 実装した改善

### 1. ログファイルを生成するテストスクリプト

`test_site_config_connection.py` を修正して、ログファイルを生成するようにしました。

### 2. 出力ラッパースクリプト

`run_test_with_output.py` を作成し、コマンド出力をファイルにリダイレクトしてから読み取るようにしました。

## 推奨される使用方法

### 方法1: ログファイルを確認

```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate
python test_site_config_connection.py 2>&1 | tee test_output.log
cat test_output.log
```

### 方法2: 直接ターミナルで実行

WSLターミナルを開いて、直接コマンドを実行してください：

```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate
python test_site_config_connection.py
```

### 方法3: 出力をファイルに保存

```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate
python test_site_config_connection.py > test_results.txt 2>&1
cat test_results.txt
```

## 今後の改善案

1. **テスト結果をJSON形式で出力**: 構造化されたデータとして出力し、後で解析しやすくする
2. **CI/CD統合**: GitHub Actionsなどで自動テストを実行し、結果を確認する
3. **ログ集約**: すべてのテスト結果を一箇所に集約する仕組みを作る

