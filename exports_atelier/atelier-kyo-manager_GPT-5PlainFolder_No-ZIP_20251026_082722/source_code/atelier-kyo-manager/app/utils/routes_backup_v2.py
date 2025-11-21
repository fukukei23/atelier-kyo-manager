# app/routes_backup_v2.py
from __future__ import annotations
from flask import Blueprint, jsonify, request
from sqlalchemy.exc import SQLAlchemyError
from app import db
try:
    from app.models import Product
except Exception:
    Product = None  # type: ignore

bp = Blueprint("backup_v2", __name__, url_prefix="/api")

@bp.get("/healthz")
def healthz():
    return jsonify({"status": "ok"}), 200

@bp.get("/products")
def list_products():
    if Product is None:
        return jsonify([]), 200
    rows = Product.query.order_by(Product.id.desc()).all()
    return jsonify([{"id": p.id, "name": p.name} for p in rows])

@bp.post("/products")
def create_product():
    if Product is None:
        return jsonify({"error": "Product model not available"}), 400
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name is required"}), 400
    try:
        p = Product(name=name)
        db.session.add(p)
        db.session.commit()
        return jsonify({"id": p.id, "name": p.name}), 201
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@bp.delete("/products/<int:product_id>")
def delete_product(product_id: int):
    if Product is None:
        return jsonify({"error": "Product model not available"}), 400
    p = Product.query.get_or_404(product_id)
    try:
        db.session.delete(p)
        db.session.commit()
        return jsonify({"status": "deleted"}), 200
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500
