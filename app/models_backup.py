# app/models.py
# -----------------------------------------------------------------------------
# SQLAlchemy モデル
# - JSON を SQLite でも安全に扱うための TypeDecorator（JSONEncodedDict）
# - Product: 取得した商品データ
# - ImageAsset: ダウンロード/生成した画像ファイルの管理
# - CrawlJob: クローリング実行の履歴とメトリクス
#
# 使い方（__init__.py 側）:
#   from .models import db
#   db.init_app(app)
#
# マイグレーション:
#   flask db init
#   flask db migrate -m "init"
#   flask db upgrade
# -----------------------------------------------------------------------------

from __future__ import annotations
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from sqlalchemy.orm import validates
from sqlalchemy.types import TypeDecorator, TEXT


db = SQLAlchemy()


# === SQLite でも JSON を扱いやすくするための TypeDecorator ====================
class JSONEncodedDict(TypeDecorator):
    """
    DB 上は TEXT として保存し、アプリでは Python の dict/list として扱う。
    - None は None のまま保存/取得
    - 文字列であっても JSON として読み出し可能なら Python オブジェクトに復元
    """
    impl = TEXT
    cache_ok = True

    def process_bind_param(self, value: Any, dialect) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        # 文字列が来たら、そのまま保存（JSON っぽくても検証しない）
        return str(value)

    def process_result_value(self, value: Optional[str], dialect) -> Any:
        if value is None:
            return None
        # 可能なら JSON として復元
        try:
            return json.loads(value)
        except Exception:
            return value


# === Product ================================================================
class Product(db.Model):
    """
    取得した商品の論理モデル
    - name / price はサイト直出しの文字列をそのまま保存（後で整形可能）
    - image_urls は重複を避けた配列（アプリ側でユニーク化推奨）
    - raw_payload は抽出元 JSON-LD 等の生データ（トラブルシュート/再抽出用）
    """
    __tablename__ = "products"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False, index=True)
    price = db.Column(db.String(64), nullable=True)
    source_url = db.Column(db.String(2000), nullable=True, unique=True, index=True)

    image_urls = db.Column(JSONEncodedDict, nullable=True)  # list[str] 想定
    raw_payload = db.Column(JSONEncodedDict, nullable=True)  # dict 想定

    note = db.Column(db.String(512), nullable=True)  # 手書きメモ

    created_at = db.Column(db.DateTime, nullable=False, server_default=func.now())
    updated_at = db.Column(
        db.DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # リレーション（画像は複数紐づく）
    assets = db.relationship(
        "ImageAsset",
        backref="product",
        lazy="dynamic",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Product id={self.id} name={self.name!r}>"

    @validates("image_urls")
    def _validate_image_urls(self, key, value):
        # None か list を許可（list の中身は文字列）
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError("image_urls must be a list")
        return [str(u) for u in value]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "price": self.price,
            "source_url": self.source_url,
            "image_urls": self.image_urls or [],
            "note": self.note,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# === ImageAsset =============================================================
class ImageAsset(db.Model):
    """
    ダウンロード/生成した画像ファイルを管理
    - original_url: 取得元URL（無い場合は None）
    - local_path: 保存した元画像のパス
    - bg_removed_path: 背景除去後のパス
    - status: 'pending' | 'processed' | 'failed'
    - meta: 幅・高さ・処理ログ等の任意情報
    """
    __tablename__ = "image_assets"

    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(
        db.Integer, db.ForeignKey("products.id", ondelete="SET NULL"), nullable=True, index=True
    )

    original_url = db.Column(db.String(2000), nullable=True)
    local_path = db.Column(db.String(1024), nullable=True)
    bg_removed_path = db.Column(db.String(1024), nullable=True)

    width = db.Column(db.Integer, nullable=True)
    height = db.Column(db.Integer, nullable=True)

    status = db.Column(db.String(32), nullable=False, default="pending", index=True)
    meta = db.Column(JSONEncodedDict, nullable=True)

    created_at = db.Column(db.DateTime, nullable=False, server_default=func.now())
    updated_at = db.Column(
        db.DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"<ImageAsset id={self.id} status={self.status} path={self.local_path!r}>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "product_id": self.product_id,
            "original_url": self.original_url,
            "local_path": self.local_path,
            "bg_removed_path": self.bg_removed_path,
            "width": self.width,
            "height": self.height,
            "status": self.status,
            "meta": self.meta or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# === CrawlJob ===============================================================
class CrawlJob(db.Model):
    """
    クローラーの実行履歴
    - target_url: クロール対象ページ
    - master_image_url: 照合用画像URL（任意）
    - status: 'pending' | 'running' | 'succeeded' | 'failed'
    - result_product_id: 成功時に紐づく Product.id
    - metrics: 計測情報（所要時間/候補件数/エラー要因などをJSONで）
    """
    __tablename__ = "crawl_jobs"

    id = db.Column(db.Integer, primary_key=True)
    target_url = db.Column(db.String(2000), nullable=False, index=True)
    master_image_url = db.Column(db.String(2000), nullable=True)
    status = db.Column(db.String(32), nullable=False, default="pending", index=True)

    result_product_id = db.Column(
        db.Integer, db.ForeignKey("products.id", ondelete="SET NULL"), nullable=True
    )
    result_product = db.relationship("Product", foreign_keys=[result_product_id])

    log_path = db.Column(db.String(1024), nullable=True)
    metrics = db.Column(JSONEncodedDict, nullable=True)

    created_at = db.Column(db.DateTime, nullable=False, server_default=func.now())
    started_at = db.Column(db.DateTime, nullable=True)
    finished_at = db.Column(db.DateTime, nullable=True)

    def __repr__(self) -> str:
        return f"<CrawlJob id={self.id} status={self.status} target={self.target_url!r}>"

    def mark_started(self) -> None:
        self.status = "running"
        self.started_at = datetime.utcnow()

    def mark_success(self, product_id: int, metrics: Optional[dict] = None) -> None:
        self.status = "succeeded"
        self.result_product_id = product_id
        self.finished_at = datetime.utcnow()
        if metrics:
            self.metrics = (self.metrics or {}) | dict(metrics)

    def mark_failed(self, metrics: Optional[dict] = None) -> None:
        self.status = "failed"
        self.finished_at = datetime.utcnow()
        if metrics:
            self.metrics = (self.metrics or {}) | dict(metrics)
