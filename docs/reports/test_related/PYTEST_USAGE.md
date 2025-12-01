# pytest 実行方法

## 概要

ターミナル実行の問題を回避するため、Python スクリプトから直接 pytest を実行できるようにしました。

## 実行方法

### 1. すべてのテストを実行

```bash
python run_pytest_direct.py
```

または

```bash
python execute_pytest_now.py
```

### 2. 特定のテストファイルを実行

```bash
python run_pytest_direct.py tests/test_session_manager.py
```

または

```bash
python execute_pytest_now.py tests/test_session_manager.py
```

### 3. 特定のテストディレクトリを実行

```bash
python run_pytest_direct.py tests/
```

### 4. 追加の pytest 引数を指定

```bash
python run_pytest_direct.py tests/test_session_manager.py -x --tb=short
```

## スクリプトの違い

- **`run_pytest_direct.py`**: フル機能版（引数解析、追加オプション対応）
- **`execute_pytest_now.py`**: シンプル版（即座に実行）

## 動作確認

Stage 1.2 のテストを実行する場合：

```bash
python execute_pytest_now.py tests/test_session_manager.py
```

## トラブルシューティング

### 仮想環境が見つからない場合

スクリプトは以下の順序で仮想環境を探します：
1. `venv/bin/python3`
2. `venv/bin/python`
3. `.venv/bin/python3`
4. `.venv/bin/python`
5. `myenv/Scripts/python.exe`

見つからない場合は、システムの Python を使用します。

### pytest が見つからない場合

仮想環境に pytest がインストールされていることを確認してください：

```bash
# 仮想環境を有効化して
source venv/bin/activate  # Linux/WSL
# または
.venv\Scripts\activate  # Windows

# pytest をインストール
pip install pytest
```

