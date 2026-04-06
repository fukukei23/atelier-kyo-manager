# ======================================================================
# プロジェクト : atelier-kyo-manager
# パッケージ   : app.routes
# 目的         : 機能別Blueprint分割のエントリポイント
# ======================================================================
from __future__ import annotations

from flask import Blueprint

bp = Blueprint("main", __name__)

# 各サブモジュールをインポート（bp にルートが登録される）
from . import products  # noqa: F401  — F01商品管理 + F02BUYMA拡張 + CSV
from . import orders    # noqa: F401  — F05注文管理 + F08キャッシュフロー
from . import partners  # noqa: F401  — F06パートナー + F13リピーター
from . import analytics # noqa: F401  — F03/F04/F07/F09-F12 各種ダッシュボード
from . import misc      # noqa: F401  — 互換リダイレクト + 自動リサーチ + API倉庫
