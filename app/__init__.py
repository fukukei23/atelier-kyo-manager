# ======================================================================
# プロジェクト: atelier-kyo-manager
# ファイル名  : app/__init__.py
# 目的        : product版では Web機能を除外（utils 実行のみ対応）
# 重要        : import 時に Flask/Web/DB を import しない（副作用ゼロ）
# 日付        : 2025-01-11 (JST)
# 備考        : product版では create_app は使用不可（RuntimeError）
# ======================================================================

from __future__ import annotations

# バージョン情報（定数のみ）
__version__ = "1.0.0"


def create_app(config_name: str | None = None):
    """
    product版では Web機能は除外されています。
    
    create_app を呼び出すと RuntimeError が発生します。
    product版では app.utils モジュールのみを使用してください。
    """
    raise RuntimeError(
        "product版ではWeb機能は除外されています。"
        " create_app は使用できません。"
        " app.utils モジュールのみを使用してください。"
    )
