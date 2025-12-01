# NexusCore プロジェクト: テスト結果自動保存機能の設定指示書

## 概要

この指示書に従って、NexusCore プロジェクトに pytest のテスト結果を自動的にファイルに保存する機能を追加してください。

atelier-kyo-manager で実装済みの機能と同様の仕組みを NexusCore にも導入します。

---

## 実装する機能

pytest を実行すると、自動的にテスト結果が `docs/reports/` ディレクトリに保存されます。

- **手動操作不要**: pytest 実行時、自動的に結果ファイルが生成される
- **タイムスタンプ付き**: 毎回新しいファイルが作成される（上書きされない）
- **詳細な情報**: 実行時間、テスト統計、失敗テストの詳細を含む

---

## 実装手順

### Step 1: `tests/conftest.py` ファイルの作成

`tests/` ディレクトリに `conftest.py` ファイルを作成し、以下の内容を追加してください。

**ファイルパス**: `tests/conftest.py`

```python
# -*- coding: utf-8 -*-
"""
pytest 設定とフック
- テスト結果を自動的にファイルに保存
"""

import pytest
from datetime import datetime
from pathlib import Path
import sys


def pytest_configure(config):
    """pytest設定時に実行されるフック"""
    # プロジェクトルートを取得
    project_root = Path(__file__).parent.parent
    reports_dir = project_root / "docs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # テスト結果ファイルのパスを設定
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = reports_dir / f"TEST_RESULTS_{timestamp}.txt"
    config._test_result_file = result_file
    
    # メタデータを保存
    config._test_start_time = datetime.now()
    
    # テスト結果を収集するためのリストを初期化
    config._test_results = []


def pytest_runtest_logreport(report):
    """各テストの結果を記録"""
    # テスト結果を収集（callフェーズのみ）
    if report.when == 'call' and hasattr(report, 'nodeid'):
        config = report.config
        if not hasattr(config, '_test_results'):
            config._test_results = []
        
        longrepr_text = None
        if hasattr(report, 'longreprtext') and report.longreprtext:
            longrepr_text = report.longreprtext
        elif hasattr(report, 'longrepr') and report.longrepr:
            # longreprtextが使えない場合はlongreprを文字列化
            longrepr_text = str(report.longrepr)
        
        config._test_results.append({
            'nodeid': report.nodeid,
            'outcome': report.outcome,  # passed, failed, skipped
            'longrepr': longrepr_text,
        })


def pytest_sessionfinish(session, exitstatus):
    """テストセッション終了時に実行されるフック"""
    if not hasattr(session.config, '_test_result_file'):
        return
    
    result_file = session.config._test_result_file
    start_time = session.config._test_start_time
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # テスト結果を集計（callフェーズのみ収集済み）
    test_results = getattr(session.config, '_test_results', [])
    
    total_tests = len(test_results)
    passed = len([r for r in test_results if r.get('outcome') == 'passed'])
    failed = len([r for r in test_results if r.get('outcome') == 'failed'])
    skipped = len([r for r in test_results if r.get('outcome') == 'skipped'])
    
    # 収集されたテスト数も記録
    collected = session.testscollected if hasattr(session, 'testscollected') else total_tests
    
    # 結果をファイルに書き込む
    try:
        with open(result_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("テスト実行結果\n")
            f.write(f"実行日時: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"終了日時: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"実行時間: {duration:.2f}秒\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"収集されたテスト数: {collected}\n")
            f.write(f"実行されたテスト数: {total_tests}\n")
            f.write(f"✅ 成功: {passed}\n")
            f.write(f"❌ 失敗: {failed}\n")
            f.write(f"⏭️  スキップ: {skipped}\n")
            f.write(f"終了コード: {exitstatus}\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
            # 失敗したテストの詳細
            if failed > 0:
                f.write("失敗したテスト:\n")
                f.write("-" * 80 + "\n")
                for result in test_results:
                    if result.get('outcome') == 'failed':
                        f.write(f"\n{result.get('nodeid', 'unknown')}\n")
                        if result.get('longrepr'):
                            f.write(f"{result.get('longrepr')}\n")
                        f.write("-" * 80 + "\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("詳細なログは pytest の出力を参照してください。\n")
            f.write(f"このファイル: {result_file}\n")
            f.write("=" * 80 + "\n")
        
        print(f"\n✅ テスト結果を保存しました: {result_file}")
        
    except Exception as e:
        print(f"\n⚠️ テスト結果ファイルの保存に失敗しました: {e}", file=sys.stderr)
```

---

### Step 2: `pytest.ini` ファイルの作成（オプション）

プロジェクトルートに `pytest.ini` ファイルを作成し、pytest の基本設定を追加してください。

**ファイルパス**: `pytest.ini`（プロジェクトルート）

```ini
[pytest]
# pytest 設定ファイル

# テスト検索パターン
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# 出力オプション
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings

# マーカー定義
markers =
    asyncio: 非同期テスト
    integration: 統合テスト
    unit: ユニットテスト
    slow: 実行時間が長いテスト
    skip: スキップするテスト

# テスト検索ディレクトリ
testpaths = tests

# ログ設定（オプション）
log_cli = false
log_cli_level = INFO
log_cli_format = %(asctime)s [%(levelname)8s] %(message)s
log_cli_date_format = %Y-%m-%d %H:%M:%S
```

**注意**: 既存の `pytest.ini` や `pyproject.toml` に pytest 設定がある場合は、`conftest.py` だけを追加すれば動作します。

---

### Step 3: `docs/reports/` ディレクトリの確認

`docs/reports/` ディレクトリが存在することを確認してください。存在しない場合は作成してください。

**動作**: `conftest.py` 内で自動的にディレクトリが作成されるため、手動で作成する必要はありません。

---

### Step 4: README ファイルの作成（オプション）

テスト結果ファイルについて説明する README ファイルを作成してください。

**ファイルパス**: `docs/reports/README_AUTO_TEST_RESULTS.md`

```markdown
# テスト結果の自動保存について

## ✅ 自動保存機能が有効になりました

このプロジェクトでは、**pytest を実行すると自動的にテスト結果がファイルに保存されます**。

## 📁 保存先とファイル名

- **ディレクトリ**: `docs/reports/`
- **ファイル名**: `TEST_RESULTS_YYYYMMDD_HHMMSS.txt`
- **例**: `TEST_RESULTS_20250128_143025.txt`

## 📋 保存される内容

1. **実行情報**
   - 実行日時
   - 終了日時
   - 実行時間（秒）

2. **テスト統計**
   - 収集されたテスト数
   - 実行されたテスト数
   - 成功数 ✅
   - 失敗数 ❌
   - スキップ数 ⏭️
   - 終了コード

3. **失敗テストの詳細**
   - 失敗したテストの名前
   - エラーメッセージとトレースバック

## 🚀 使い方

**通常通り pytest を実行するだけです！**

```bash
# すべてのテストを実行
pytest

# 特定のテストファイルを実行
pytest tests/test_example.py

# 詳細な出力付きで実行
pytest -v

# 長いトレースバックで実行
pytest --tb=long
```

実行後、`docs/reports/` ディレクトリに結果ファイルが自動的に保存されます。

## 🔧 実装詳細

自動保存機能は以下のファイルで実装されています：

- **`tests/conftest.py`**: pytest フックでテスト結果を自動収集・保存
- **`pytest.ini`**: pytest の基本設定（オプション）

## 📝 注意事項

1. **ファイルは上書きされません**
   - タイムスタンプ付きのファイル名なので、毎回新しいファイルが作成されます

2. **古いファイルの削除**
   - 必要に応じて手動で削除してください

3. **.gitignore**
   - テスト結果ファイルは Git に含めても構いませんが、大量になる場合は `.gitignore` に追加することを検討してください

## 🐛 トラブルシューティング

### 結果ファイルが生成されない場合

1. `tests/conftest.py` が存在することを確認
2. pytest が正常に終了していることを確認（途中で中断された場合は保存されない可能性があります）
3. `docs/reports/` ディレクトリに書き込み権限があることを確認

### 結果ファイルの内容が不完全な場合

- テストが途中で中断された可能性があります
- エラーが発生した場合は、ターミナル出力も確認してください
```

---

## 実装確認手順

### 1. ファイルの作成確認

以下のファイルが作成されていることを確認してください：

- ✅ `tests/conftest.py`
- ✅ `pytest.ini`（オプション）
- ✅ `docs/reports/README_AUTO_TEST_RESULTS.md`（オプション）

### 2. 動作確認

簡単なテストを実行して、結果ファイルが生成されることを確認してください：

```bash
# プロジェクトルートで実行
pytest

# または、特定のテストファイルを実行
pytest tests/test_example.py -v
```

### 3. 結果ファイルの確認

実行後、以下のメッセージが表示されることを確認してください：

```
✅ テスト結果を保存しました: docs/reports/TEST_RESULTS_YYYYMMDD_HHMMSS.txt
```

そして、`docs/reports/` ディレクトリに結果ファイルが生成されていることを確認してください。

---

## 既存プロジェクトへの対応

### 既に `tests/conftest.py` が存在する場合

既存の `conftest.py` に、以下の関数を追加してください：

1. `pytest_configure(config)` - 既存の関数がある場合は、内容をマージ
2. `pytest_runtest_logreport(report)` - 新しい関数として追加
3. `pytest_sessionfinish(session, exitstatus)` - 新しい関数として追加

**注意**: 既存の関数と名前が重複する場合は、内容をマージしてください。

### 既に `pytest.ini` が存在する場合

既存の `pytest.ini` はそのまま使用できます。`conftest.py` だけを追加すれば動作します。

---

## ファイル構造

実装後のプロジェクト構造：

```
NexusCore/
├── pytest.ini              # pytest 設定（新規作成）
├── tests/
│   ├── conftest.py         # pytest フック（新規作成）
│   └── test_*.py           # 既存のテストファイル
└── docs/
    └── reports/
        ├── README_AUTO_TEST_RESULTS.md  # 説明ファイル（オプション）
        └── TEST_RESULTS_*.txt           # 自動生成される結果ファイル
```

---

## トラブルシューティング

### 問題: 結果ファイルが生成されない

**解決策**:
1. `tests/conftest.py` が正しい場所に作成されているか確認
2. pytest が正常に終了しているか確認（Ctrl+C で中断した場合は保存されません）
3. `docs/reports/` ディレクトリに書き込み権限があるか確認

### 問題: エラーが発生する

**解決策**:
1. `conftest.py` のインデントや構文エラーを確認
2. pytest のバージョンが最新か確認（`pip install --upgrade pytest`）
3. 既存の `conftest.py` との競合を確認

### 問題: 既存のテストが動作しなくなる

**解決策**:
- `conftest.py` は pytest の標準的な拡張方法なので、既存のテストには影響しません
- 問題が発生した場合は、既存の `conftest.py` との競合を確認してください

---

## まとめ

この指示書に従って実装することで、NexusCore プロジェクトでも atelier-kyo-manager と同様に、テスト結果が自動的にファイルに保存されるようになります。

**実装のポイント**:
1. `tests/conftest.py` を追加（必須）
2. `pytest.ini` を追加（推奨、オプション）
3. テストを実行して動作確認

**質問や問題がある場合は、atelier-kyo-manager の実装（`tests/conftest.py` と `pytest.ini`）を参考にしてください。**

---

**作成日**: 2025-01-28  
**参照元**: atelier-kyo-manager の実装

