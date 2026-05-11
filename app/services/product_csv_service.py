"""Product CSV import/export service."""

from __future__ import annotations

import csv
import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from app.models import Product

EXPORT_HEADERS = [
    "id",
    "name",
    "brand",
    "purchase_price",
    "selling_price",
    "transaction_fee",
    "shipping_cost",
    "customs_duty",
    "procurement_fee",
    "supplier_url",
    "image_url",
    "stock_status",
    "brand_tier",
    "source_type",
    "source_region",
    "color",
    "size",
    "material",
    "description",
    "retail_price",
    "target_profit_rate",
    "listing_status",
    "created_at",
    "updated_at",
    "warehouse_shipping_cost",
    "original_currency",
    "exchange_rate",
    "item_category",
]

CANDIDATE_HEADERS = [
    "id",
    "name",
    "brand",
    "brand_tier",
    "purchase_price",
    "selling_price",
    "profit",
    "profit_rate",
    "stock_status",
    "listing_status",
    "target_profit_rate",
    "source_type",
    "item_category",
]


def _row_to_product(row: dict[str, str]) -> Product:
    """CSV行からProductインスタンスを生成する."""
    from app.models import Product

    return Product(
        name=row.get("name"),
        brand=row.get("brand"),
        purchase_price=float(row.get("purchase_price", 0) or 0),
        selling_price=float(row.get("selling_price", 0) or 0),
        transaction_fee=float(row.get("transaction_fee", 0) or 0),
        shipping_cost=float(row.get("shipping_cost", 0) or 0),
        customs_duty=float(row.get("customs_duty", 0) or 0),
        procurement_fee=float(row.get("procurement_fee", 0) or 0),
        supplier_url=row.get("supplier_url"),
        image_url=row.get("image_url"),
        stock_status=str(row.get("stock_status", "")).strip() in ("1", "true", "True", "yes", "on"),
        source_type=row.get("source_type") or None,
        source_region=row.get("source_region") or None,
        color=row.get("color") or None,
        size=row.get("size") or None,
        material=row.get("material") or None,
        description=row.get("description") or None,
        retail_price=float(row.get("retail_price", 0) or 0) or None,
        target_profit_rate=float(row.get("target_profit_rate", 10) or 10) / 100.0,
        listing_status=row.get("listing_status") or "draft",
        warehouse_shipping_cost=float(row.get("warehouse_shipping_cost", 0) or 0),
        original_currency=row.get("original_currency", "JPY") or "JPY",
        exchange_rate=float(row.get("exchange_rate", 1) or 1),
        item_category=row.get("item_category") or None,
    )


def import_products_from_csv(file_content: str, db_session: Session) -> int:
    """CSV文字列からProductを一括インポートし、件数を返す."""
    reader = csv.DictReader(io.StringIO(file_content))
    count = 0
    for row in reader:
        if not row.get("name"):
            continue
        if row.get("purchase_price") in (None, "") or row.get("selling_price") in (None, ""):
            continue
        p = _row_to_product(row)
        p.auto_classify_tier()
        db_session.add(p)
        count += 1
    db_session.commit()
    return count


def _product_to_row(p: Product) -> dict[str, str | int | float]:
    """ProductインスタンスをCSV行dictに変換する."""
    return {
        "id": p.id,
        "name": p.name,
        "brand": p.brand,
        "purchase_price": p.purchase_price,
        "selling_price": p.selling_price,
        "transaction_fee": p.transaction_fee,
        "shipping_cost": p.shipping_cost,
        "customs_duty": p.customs_duty,
        "procurement_fee": p.procurement_fee,
        "supplier_url": p.supplier_url,
        "image_url": p.image_url,
        "stock_status": int(bool(p.stock_status)),
        "brand_tier": p.brand_tier or "",
        "source_type": p.source_type or "",
        "source_region": p.source_region or "",
        "color": p.color or "",
        "size": p.size or "",
        "material": p.material or "",
        "description": p.description or "",
        "retail_price": p.retail_price or "",
        "target_profit_rate": round((p.target_profit_rate or 0) * 100, 1),
        "listing_status": p.listing_status or "draft",
        "created_at": (p.created_at.isoformat(sep=" ", timespec="seconds") if p.created_at else ""),
        "updated_at": (p.updated_at.isoformat(sep=" ", timespec="seconds") if p.updated_at else ""),
        "warehouse_shipping_cost": p.warehouse_shipping_cost or 0,
        "original_currency": p.original_currency or "JPY",
        "exchange_rate": p.exchange_rate or 1.0,
        "item_category": p.item_category or "",
    }


def export_products_csv(products: list[Product]) -> str:
    """ProductリストをCSV文字列に変換する."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=EXPORT_HEADERS)
    writer.writeheader()
    for p in products:
        writer.writerow(_product_to_row(p))
    return buf.getvalue()


def _candidate_to_row(p: Product) -> dict[str, str | int | float]:
    """出品候補ProductをCSV行dictに変換する."""
    return {
        "id": p.id,
        "name": p.name,
        "brand": p.brand or "",
        "brand_tier": p.brand_tier or "",
        "purchase_price": p.purchase_price,
        "selling_price": p.selling_price,
        "profit": round(p.calculate_profit(), 0),
        "profit_rate": round(p.profit_rate(), 1),
        "stock_status": int(bool(p.stock_status)),
        "listing_status": p.listing_status or "draft",
        "target_profit_rate": round((p.target_profit_rate or 0) * 100, 1),
        "source_type": p.source_type or "",
        "item_category": p.item_category or "",
    }


def export_candidates_csv(candidates: list[Product]) -> str:
    """出品候補リストをCSV文字列に変換する."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CANDIDATE_HEADERS)
    writer.writeheader()
    for p in candidates:
        writer.writerow(_candidate_to_row(p))
    return buf.getvalue()
