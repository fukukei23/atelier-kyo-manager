# WSL環境でのコマンド出力取得の改善

## 問題の原因

WSL環境で `run_terminal_cmd` ツールを使用しても、コマンド出力が取得できません。これは以下の制限によるものです：

1. **WSLとCursorの統合制限**: WSL環境でのコマンド実行時に、標準出力が正しくキャプチャされない可能性
2. **バッファリング**: Pythonの出力バッファリングにより、出力が即座に表示されない
3. **シェルの設定**: シェルの設定により、出力がリダイレクトされている

## 実装した改善

### 1. ログファイルを生成するテストスクリプト

`test_site_config_connection.py` を修正して、ログファイル (`test_site_config_connection.log`) を生成するようにしました。

### 2. 出力ラッパースクリプト

`run_test_with_output.py` を作成し、コマンド出力をファイルにリダイレクトしてから読み取るようにしました。

### 3. 結果確認スクリプト

`check_test_results.sh` を作成し、テスト結果を確認しやすくしました。

## 使用方法

### 方法1: ログファイルを確認（推奨）

```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate
python test_site_config_connection.py
# 実行後、ログファイルを確認
cat test_site_config_connection.log
```

### 方法2: 出力をファイルにリダイレクト

```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate
python test_site_config_connection.py > test_output.log 2>&1
cat test_output.log
```

### 方法3: 結果確認スクリプトを使用

```bash
cd /home/yn441611/atelier-kyo-manager
./check_test_results.sh
```

### 方法4: 直接ターミナルで実行（最も確実）

WSLターミナルを開いて、直接コマンドを実行してください：

```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate
python test_site_config_connection.py
```

## 今後の改善案

1. **テスト結果をJSON形式で出力**: 構造化されたデータとして出力し、後で解析しやすくする
2. **CI/CD統合**: GitHub Actionsなどで自動テストを実行し、結果を確認する
3. **ログ集約**: すべてのテスト結果を一箇所に集約する仕組みを作る
4. **リアルタイムログ表示**: ログファイルを監視して、リアルタイムで表示する仕組みを作る

## 結論

WSL環境でのコマンド出力取得は、Cursorの制限により完全には解決できませんが、以下の方法で対処できます：

1. **ログファイルを生成**: テストスクリプト自体がログファイルを生成する
2. **ファイルにリダイレクト**: コマンド実行時に出力をファイルにリダイレクト
3. **直接ターミナルで実行**: 最も確実な方法

これらの方法により、テスト結果を確認できるようになります。

