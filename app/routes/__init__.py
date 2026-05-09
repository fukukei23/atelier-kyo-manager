# ======================================================================
# プロジェクト : atelier-kyo-manager
# パッケージ   : app.routes
# 目的         : 機能別Blueprint分割のエントリポイント
# ======================================================================
from __future__ import annotations

from flask import Blueprint

bp = Blueprint("main", __name__)

# 各サブモジュールをインポート（bp にルートが登録される）
from . import (
    analytics,  # noqa: F401  — F09ブランド分析 + FR-018ダッシュボード
    auto_orders,  # noqa: F401  — FR-009 AI自動発注管理
    chatbot,  # noqa: F401  — FR-010顧客対応AI ChatBot
    faq_templates,  # noqa: F401  — FR-010基盤FAQテンプレート管理
    listing_progress,  # noqa: F401  — F07品出し進捗トラッカー
    listing_templates,  # noqa: F401  — F03出品テンプレート管理
    misc,  # noqa: F401  — 互換リダイレクト + 自動リサーチ + API倉庫
    orders,  # noqa: F401  — F05注文管理 + F08キャッシュフロー
    partners,  # noqa: F401  — F06パートナー + F13リピーター
    popularity,  # noqa: F401  — F11人気度トラッキング
    products,  # noqa: F401  — F01商品管理 + F02BUYMA拡張 + CSV
    prohibited_sources,  # noqa: F401  — F04禁制品買付先チェックAPI
    region_recommendations,  # noqa: F401  — F12買付先地域レコメンド
    shipment_notifications,  # noqa: F401  — FR-012基盤発送通知管理
    stock_checks,  # noqa: F401  — F10在庫＆価格チェック
)
