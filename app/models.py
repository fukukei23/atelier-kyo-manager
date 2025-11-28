# ======================================================================
# プロジェクト: atelier-kyo-manager
# ファイル名  : app/models.py   ← 全置換
# 目的        : DBの実スキーマ（site.db の product テーブル）に合わせたモデル定義
# 日付        : 2025-08-23 (JST)
# 注意        : DBに存在しないカラム（price/source_url/image_urls/note）は削除
#               manage.html で使う列をすべて含め、ズレを解消
# ======================================================================

from __future__ import annotations
from datetime import datetime

from sqlalchemy import Integer, String, Float, Boolean, DateTime, Text
from .extensions import db


class Product(db.Model):
    __tablename__ = "product"

    id = db.Column(Integer, primary_key=True)
    name = db.Column(String(255), nullable=False)

    # manage.html / 実DBに存在する列
    brand = db.Column(String(128), nullable=True)
    purchase_price = db.Column(Float, nullable=False)     # PRAGMA: NOT NULL
    selling_price = db.Column(Float, nullable=False)      # PRAGMA: NOT NULL
    supplier_url = db.Column(String(512), nullable=True)
    image_url = db.Column(String(512), nullable=True)
    stock_status = db.Column(Boolean, default=False, nullable=True)
    profit = db.Column(Float, nullable=True)              # 表示用に残存（計算メソッドも提供）
    transaction_fee = db.Column(Float, nullable=True)
    shipping_cost = db.Column(Float, nullable=True)
    customs_duty = db.Column(Float, nullable=True)
    procurement_fee = db.Column(Float, nullable=True)

    created_at = db.Column(DateTime, default=datetime.utcnow, nullable=True)
    updated_at = db.Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=True,
    )

    def calculate_profit(self) -> float:
        """
        利益の簡易計算:
          販売価格 - (仕入れ + 取引手数料 + 送料 + 関税 + 買付代行料)
        DBの profit カラムがあっても、表示は都度計算で返します。
        """
        nz = lambda x: float(x or 0.0)
        selling = nz(self.selling_price)
        costs = (
            nz(self.purchase_price)
            + nz(self.transaction_fee)
            + nz(self.shipping_cost)
            + nz(self.customs_duty)
            + nz(self.procurement_fee)
        )
        return float(selling - costs)

    def __repr__(self) -> str:
        return f"<Product id={self.id} name={self.name!r}>"


class ChatHistory(db.Model):
    """
    チャット履歴を永続化するモデル
    再起動後もチャット履歴が保持されるようにする
    """
    __tablename__ = "chat_history"

    id = db.Column(Integer, primary_key=True)
    session_id = db.Column(String(255), nullable=False, index=True)  # セッション識別子
    role = db.Column(String(50), nullable=False)  # "user" or "assistant"
    content = db.Column(Text, nullable=False)  # メッセージ内容
    model_family = db.Column(String(50), nullable=True)  # 使用したモデル（gemini, openai等）
    tokens = db.Column(Integer, nullable=True)  # 使用トークン数
    cost_usd = db.Column(Float, nullable=True)  # コスト（USD）
    created_at = db.Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    def __repr__(self) -> str:
        return f"<ChatHistory id={self.id} session_id={self.session_id!r} role={self.role!r}>"
