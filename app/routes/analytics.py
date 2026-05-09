# ======================================================================
# F09: ブランド分析 + FR-018: Analyticsダッシュボード
# ======================================================================
from __future__ import annotations

import csv
from datetime import datetime, timedelta
from io import StringIO

from flask import Response, render_template, request
from flask_login import login_required
from sqlalchemy import func

from app.extensions import db
from app.models import Product

from . import bp


@bp.get("/brand-analytics")
@login_required
def brand_analytics():
    """ブランド階層別利益率ダッシュボード"""
    tier_stats = (
        db.session.query(
            Product.brand_tier,
            func.count(Product.id).label("count"),
            func.avg(Product.target_profit_rate).label("avg_target_rate"),
            func.avg(Product.selling_price).label("avg_selling"),
            func.avg(Product.purchase_price).label("avg_cost"),
        )
        .group_by(Product.brand_tier)
        .all()
    )

    tier_data = {}
    for ts in tier_stats:
        tier = ts.brand_tier or "low"
        selling = float(ts.avg_selling or 0)
        cost = float(ts.avg_cost or 0)
        profit = selling - cost
        rate = (profit / selling * 100) if selling > 0 else 0
        tier_data[tier] = {
            "count": ts.count,
            "avg_selling": round(selling, 0),
            "avg_cost": round(cost, 0),
            "avg_profit": round(profit, 0),
            "avg_profit_rate": round(rate, 1),
            "avg_target_rate": round(float(ts.avg_target_rate or 0) * 100, 1),
        }

    products = Product.query.order_by(Product.brand_tier, Product.id.desc()).all()
    return render_template("brand_analytics.html", tier_data=tier_data, products=products)


@bp.get("/dashboard")
@login_required
def dashboard():
    """Analyticsダッシュボード: KPI + パイプライン + 在庫 + ブランド階層"""
    query = Product.query
    period = request.args.get("period", "")
    if period == "7d":
        query = query.filter(Product.created_at >= datetime.utcnow() - timedelta(days=7))
    elif period == "30d":
        query = query.filter(Product.created_at >= datetime.utcnow() - timedelta(days=30))
    brand = request.args.get("brand", "")
    if brand:
        query = query.filter(Product.brand == brand)

    # CSVエクスポート
    if request.args.get("export") == "csv":
        products = query.order_by(Product.created_at.desc()).all()
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(
            ["商品名", "ブランド", "仕入価格", "販売価格", "利益率", "パイプライン状態", "在庫", "出品状態"]
        )
        for p in products:
            margin = ""
            if p.purchase_price and p.selling_price and p.purchase_price > 0:
                margin = f"{((p.selling_price - p.purchase_price) / p.purchase_price * 100):.1f}%"
            writer.writerow(
                [
                    p.name,
                    p.brand,
                    p.purchase_price,
                    p.selling_price,
                    margin,
                    p.pipeline_status or "",
                    "あり" if p.stock_status else "なし",
                    p.listing_status or "",
                ]
            )
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Response(
            output.getvalue(),
            mimetype="text/csv; charset=utf-8-sig",
            headers={"Content-Disposition": f"attachment; filename=dashboard_{ts}.csv"},
        )

    # KPI
    total_products = query.count()
    products = query.all()
    rates = [p.profit_rate() for p in products]
    profits = [p.calculate_profit() for p in products]
    kpi = {
        "total_products": total_products,
        "avg_profit_rate": sum(rates) / len(rates) if rates else 0,
        "total_profit": sum(profits),
    }

    # パイプライン進捗
    pipeline_rows = (
        db.session.query(Product.pipeline_status, func.count(Product.id)).group_by(Product.pipeline_status).all()
    )
    pipeline_summary = {"pending": 0, "running": 0, "success": 0, "partial": 0, "failed": 0}
    for status, cnt in pipeline_rows:
        key = status or "pending"
        if key in pipeline_summary:
            pipeline_summary[key] = cnt

    # 在庫ステータス
    in_stock = Product.query.filter(Product.stock_status).count()
    stock_summary = {"in_stock": in_stock, "out_of_stock": total_products - in_stock}

    # 出品ステータス
    listing_rows = (
        db.session.query(Product.listing_status, func.count(Product.id)).group_by(Product.listing_status).all()
    )
    listing_summary = {"draft": 0, "listed": 0, "sold": 0, "archived": 0}
    for status, cnt in listing_rows:
        key = status or "draft"
        if key in listing_summary:
            listing_summary[key] = cnt

    # ブランド階層別利益率
    tier_labels = []
    tier_rates = []
    for tier in ("high", "medium", "low"):
        tier_prods = [p for p in products if (p.brand_tier or "low") == tier]
        if tier_prods:
            avg = sum(p.profit_rate() for p in tier_prods) / len(tier_prods)
        else:
            avg = 0
        label_map = {"high": "ハイブランド", "medium": "ミドル", "low": "ロー"}
        tier_labels.append(label_map[tier])
        tier_rates.append(round(avg, 1))

    brands = [r[0] for r in db.session.query(Product.brand).distinct().all() if r[0]]

    return render_template(
        "dashboard.html",
        kpi=kpi,
        pipeline_summary=pipeline_summary,
        stock_summary=stock_summary,
        listing_summary=listing_summary,
        tier_labels=tier_labels,
        tier_rates=tier_rates,
        period=period,
        brand=brand,
        brands=brands,
    )
