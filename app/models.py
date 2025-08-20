# app/models.py
from __future__ import annotations
from datetime import datetime
from sqlalchemy import Integer, String, DateTime
from sqlalchemy.dialects.sqlite import JSON
from .extensions import db

class Product(db.Model):
    __tablename__ = "product"

    id = db.Column(Integer, primary_key=True)
    name = db.Column(String(255), nullable=False)
    price = db.Column(String(64), nullable=True)
    source_url = db.Column(String(1024), nullable=True)
    image_urls = db.Column(JSON().with_variant(JSON, "sqlite"), nullable=True)  # list[str] 前提
    note = db.Column(String(255), nullable=True)
    created_at = db.Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<Product id={self.id} name={self.name!r}>"
