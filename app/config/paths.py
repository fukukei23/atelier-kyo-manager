"""
プロジェクトの標準パス定義
今後のファイル生成はこのモジュールのパスを使用してください
"""

from pathlib import Path

# プロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 標準ディレクトリ
SCRIPTS_DEV_DIR = PROJECT_ROOT / "scripts" / "dev"
DOCS_REPORTS_DIR = PROJECT_ROOT / "docs" / "reports"
LOGS_GENERATED_DIR = PROJECT_ROOT / "logs" / "generated"
TMP_GENERATED_DIR = PROJECT_ROOT / "tmp" / "generated"
DATA_GENERATED_DIR = PROJECT_ROOT / "data" / "generated"

# 既存のディレクトリ（互換性のため）
INSTANCE_DIR = PROJECT_ROOT / "instance"
OUTPUT_DIR = PROJECT_ROOT / "output"
EXPORTS_DIR = PROJECT_ROOT / "exports"


def ensure_dirs():
    """必要なディレクトリを作成"""
    for dir_path in [
        SCRIPTS_DEV_DIR,
        DOCS_REPORTS_DIR,
        LOGS_GENERATED_DIR,
        TMP_GENERATED_DIR,
        DATA_GENERATED_DIR,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)
