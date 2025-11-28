# プロジェクトファイル整理ガイド

## フォルダ構造

プロジェクトルートのファイルは以下のフォルダに整理されています：

### 📁 `scripts/dev/`
開発用スクリプト（テスト、チェック、実行スクリプトなど）
- `check_*.py`
- `execute_*.py`
- `run_*_test*.py`
- `test_*.py`
- `verify_*.py`
- `auto_*.py`
- `debug_*.py`
- その他の開発用スクリプト

### 📄 `docs/reports/`
レポート・ドキュメント類
- `*_REPORT.md`
- `*_ANALYSIS.md`
- `*_SUMMARY.md`
- `*_GUIDE.md`
- `STAGE_*.md`
- `PROJECT_ANALYSIS_REPORT*.md`
- その他のレポート類

### 📋 `logs/generated/`
ログファイル
- `*.log`
- `browser_test_*.log`
- `test_*.log`

### 🗑️ `tmp/generated/`
一時ファイル
- `coverage.json`
- `depgraph.json`
- `project_dependency_graph.json`
- `*.html` (一時的なHTMLファイル)
- `*.csv` (一時的なCSVファイル)
- `instance.zip`

### 💾 `data/generated/`
生成されたデータファイル
- `catalog_*.csv`
- その他のデータファイル

## ルートに残すファイル

以下のファイルはプロジェクトルートに残します：
- `README.md`
- `requirements.txt`
- `.gitignore`
- `.gitattributes`
- `.env.template`
- `pyrightconfig.json`
- `atelier-kyo-manager.code-workspace`
- `overrides.local.json` (ローカル設定)

## 今後のファイル生成

新しいファイルを生成する際は、以下のパスを使用してください：

```python
from pathlib import Path

# プロジェクトルート
ROOT = Path(__file__).parent.parent  # app/ から見た場合
ROOT = Path(__file__).parent         # ルートから見た場合

# 開発スクリプト
DEV_SCRIPTS_DIR = ROOT / "scripts" / "dev"

# レポート
REPORTS_DIR = ROOT / "docs" / "reports"

# ログ
LOGS_DIR = ROOT / "logs" / "generated"

# 一時ファイル
TMP_DIR = ROOT / "tmp" / "generated"

# データファイル
DATA_DIR = ROOT / "data" / "generated"
```

## 整理スクリプト

ファイルを整理するには：
```bash
bash organize_files.sh
```

または：
```bash
python3 organize_project_root.py --execute
```

