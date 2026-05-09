#!/usr/bin/env python3
"""
サンプルデータ投入スクリプト（冪等性あり）

使い方:
    # 全データ投入（既存データはスキップ）
    python seed_data.py

    # データを全削除してから投入
    python seed_data.py --reset

    # 管理者ユーザーのみ作成
    python seed_data.py --admin-only --username admin --password changeme
"""
from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import create_app
from app.extensions import db
from app.models import (
    Product, Partner, Order, RepeatCustomer, ListingTemplate,
    ListingProgress, StockCheck, PopularityTracker, RegionRecommendation,
)
from app.models.user import User


# ------------------------------------------------------------------
# 定数: テストデータ
# ------------------------------------------------------------------
PRODUCTS_DATA = [
    dict(name="Classic Flap Bag", brand="Chanel", purchase_price=450000, selling_price=585000,
         supplier_url="https://example.com/chanel-001", color="Black", size="M", material="Lambskin",
         source_type="overseas", source_region="FR", retail_price=620000, item_category="bag",
         warehouse_shipping_cost=3500, original_currency="EUR", exchange_rate=165.0),
    dict(name="Birkin 25", brand="Hermes", purchase_price=850000, selling_price=1105000,
         supplier_url="https://example.com/hermes-001", color="Gold", size="25cm", material="Togo Leather",
         source_type="overseas", source_region="FR", retail_price=1200000, item_category="bag",
         warehouse_shipping_cost=4000, original_currency="EUR", exchange_rate=165.0),
    dict(name="Neverfull MM", brand="Louis Vuitton", purchase_price=140000, selling_price=182000,
         supplier_url="https://example.com/lv-001", color="Damier Azur", size="MM", material="Canvas",
         source_type="overseas", source_region="US", retail_price=165000, item_category="bag",
         warehouse_shipping_cost=2500, original_currency="USD", exchange_rate=150.0),
    dict(name="GG Marmont Small", brand="Gucci", purchase_price=180000, selling_price=234000,
         supplier_url="https://example.com/gucci-001", color="Gold", size="Small", material="Leather",
         source_type="overseas", source_region="IT", retail_price=210000, item_category="bag",
         warehouse_shipping_cost=3000, original_currency="EUR", exchange_rate=165.0),
    dict(name="Re-Edition 2005", brand="Prada", purchase_price=95000, selling_price=123500,
         supplier_url="https://example.com/prada-001", color="Black", size="One Size", material="Saffiano",
         source_type="overseas", source_region="IT", retail_price=115000, item_category="bag",
         warehouse_shipping_cost=2500, original_currency="EUR", exchange_rate=165.0),
    dict(name="Nylon Logo Jacket", brand="Moncler", purchase_price=120000, selling_price=156000,
         supplier_url="https://example.com/moncler-001", color="Navy", size="M", material="Nylon",
         source_type="overseas", source_region="IT", retail_price=145000, item_category="jacket",
         warehouse_shipping_cost=3000, original_currency="EUR", exchange_rate=165.0),
    dict(name="Air Jordan 1 Retro High", brand="Nike", purchase_price=25000, selling_price=32500,
         supplier_url="https://example.com/nike-001", color="Chicago", size="27cm", material="Leather",
         source_type="overseas", source_region="US", retail_price=30000, item_category="shoes",
         warehouse_shipping_cost=2000, original_currency="USD", exchange_rate=150.0),
    dict(name="Basic T-Shirt", brand="ZARA", purchase_price=2500, selling_price=3250,
         supplier_url="https://example.com/zara-001", color="White", size="M", material="Cotton",
         source_type="domestic", source_region="ES", item_category="tops",
         warehouse_shipping_cost=500),
    dict(name="Heattech Ultra Warm", brand="UNIQLO", purchase_price=1500, selling_price=1950,
         supplier_url="https://example.com/uniqlo-001", color="Black", size="L", material="Polyester",
         source_type="domestic", source_region="JP", item_category="tops",
         warehouse_shipping_cost=0),
    dict(name="Cotton Blazer", brand="H&M", purchase_price=4500, selling_price=5850,
         supplier_url="https://example.com/hm-001", color="Gray", size="M", material="Cotton",
         source_type="domestic", source_region="SE", item_category="jacket",
         warehouse_shipping_cost=500),
]

# 商品の stock_status / listing_status / pipeline_status
PRODUCT_FLAGS = [
    dict(stock_status=True, listing_status="listed", pipeline_status="pending"),
    dict(stock_status=True, listing_status="listed", pipeline_status="pending"),
    dict(stock_status=True, listing_status="listed", pipeline_status="success"),
    dict(stock_status=True, listing_status="listed", pipeline_status="success"),
    dict(stock_status=True, listing_status="draft", pipeline_status="pending"),
    dict(stock_status=True, listing_status="draft", pipeline_status="pending"),
    dict(stock_status=True, listing_status="listed", pipeline_status="partial"),
    dict(stock_status=True, listing_status="draft", pipeline_status="pending"),
    dict(stock_status=False, listing_status="sold", pipeline_status="pending"),
    dict(stock_status=True, listing_status="archived", pipeline_status="pending"),
]


def seed_admin(username: str = "admin", password: str = "changeme", display_name: str = "Atelier Kyo") -> None:
    """管理者ユーザー作成（冪等）"""
    existing = User.query.filter_by(username=username).first()
    if existing:
        print(f"  User '{username}' already exists (id={existing.id}). Skip.")
        return
    user = User(username=username, display_name=display_name, is_admin=True, is_active=True)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    print(f"  Created admin: {username} (password={password})")


def seed_products() -> list[Product]:
    """商品データ（冪等: name+brandで重複チェック）"""
    products = []
    for i, data in enumerate(PRODUCTS_DATA):
        flags = PRODUCT_FLAGS[i]
        exists = Product.query.filter_by(name=data["name"], brand=data["brand"]).first()
        if exists:
            products.append(exists)
            continue
        p = Product(
            **data,
            image_url=f"https://placehold.co/300x300?text={data['brand']}",
            stock_status=flags["stock_status"],
            listing_status=flags["listing_status"],
            target_profit_rate=0.25,
            pipeline_status=flags["pipeline_status"],
        )
        p.auto_classify_tier()
        db.session.add(p)
        products.append(p)
    db.session.flush()
    print(f"  Product: {len(products)}件")
    return products


def seed_partners() -> list[Partner]:
    """パートナー（冪等: nameで重複チェック）"""
    data = [
        dict(name="Global Fashion Hub", email="contact@globalfashion.com", phone="+1-212-555-0100",
             active_regions="US,CA", specialty_brands="Chanel,Louis Vuitton,Gucci",
             priority_level="high", status="active"),
        dict(name="Euro Luxe Partners", email="info@euroluxe.eu", phone="+39-02-555-0200",
             active_regions="IT,FR,DE,UK", specialty_brands="Hermes,Prada,Moncler",
             priority_level="medium", status="active"),
        dict(name="Tokyo Connect", email="sales@tokyoconnect.jp", phone="03-5555-0300",
             active_regions="JP", specialty_brands="UNIQLO,ZARA",
             priority_level="low", status="active"),
    ]
    partners = []
    for d in data:
        exists = Partner.query.filter_by(name=d["name"]).first()
        if exists:
            partners.append(exists)
            continue
        p = Partner(**d)
        db.session.add(p)
        partners.append(p)
    db.session.flush()
    print(f"  Partner: {len(partners)}件")
    return partners


def seed_orders(products: list[Product], partners: list[Partner]) -> None:
    """注文データ"""
    if Order.query.first():
        print("  Order: skip (already exists)")
        return
    today = datetime.now()
    orders = [
        Order(order_number="BY-2026-0001", product_name="Classic Flap Bag", customer_name="田中太郎",
              order_date=today - timedelta(days=5), selling_price=585000, purchase_cost=450000,
              customs_duty=45000, payment_method="credit_card", source_type="overseas",
              status="pending", product_id=products[0].id, partner_id=partners[0].id),
        Order(order_number="BY-2026-0002", product_name="GG Marmont Small", customer_name="佐藤花子",
              order_date=today - timedelta(days=12), selling_price=234000, purchase_cost=180000,
              customs_duty=18000, payment_method="rakuten_pay", source_type="overseas",
              status="shipped", product_id=products[3].id, partner_id=partners[1].id),
        Order(order_number="BY-2026-0003", product_name="Air Jordan 1 Retro High", customer_name="鈴木一郎",
              order_date=today - timedelta(days=20), selling_price=32500, purchase_cost=25000,
              customs_duty=2500, payment_method="d_pay", source_type="overseas",
              status="completed", product_id=products[6].id, partner_id=partners[0].id),
        Order(order_number="BY-2026-0004", product_name="Birkin 25", customer_name="高橋美咲",
              order_date=today - timedelta(days=3), selling_price=1105000, purchase_cost=850000,
              customs_duty=85000, payment_method="bank_transfer", source_type="overseas",
              status="pending", product_id=products[1].id, partner_id=partners[1].id),
        Order(order_number="BY-2026-0005", product_name="Re-Edition 2005", customer_name="伊藤健太",
              order_date=today - timedelta(days=15), selling_price=123500, purchase_cost=95000,
              customs_duty=9500, payment_method="credit_card", source_type="overseas",
              status="cancelled", product_id=products[4].id, partner_id=partners[1].id),
    ]
    for o in orders:
        o.calc_deadlines()
        o.calc_profit()
    db.session.add_all(orders)
    print(f"  Order: {len(orders)}件")


def seed_templates() -> None:
    """出品テンプレート"""
    if ListingTemplate.query.first():
        print("  ListingTemplate: skip (already exists)")
        return
    db.session.add_all([
        ListingTemplate(name="標準テンプレート",
                        template_text="【{{brand}}】{{productName}}\n\nカラー: {{color}}\nサイズ: {{size}}\n素材: {{material}}\n定価: {{retailPrice}}\n仕入国: {{sourceCountry}}\n状態: 新品未使用・タグ付き",
                        category="general", is_default=True),
        ListingTemplate(name="ファッション用テンプレート",
                        template_text="{{brand}} {{productName}}\n\nカラー: {{color}} / サイズ: {{size}} / 素材: {{material}}\n\n{{description}}",
                        category="fashion", is_default=False),
    ])
    print("  ListingTemplate: 2件")


def seed_remaining(products: list[Product]) -> None:
    """残りのモデル（RepeatCustomer, ListingProgress, StockCheck, Popularity, Region）"""
    today = datetime.now()

    # RepeatCustomer
    if not RepeatCustomer.query.first():
        customers = [
            RepeatCustomer(customer_name="田中太郎", email="tanaka@example.com", phone="090-1111-2222",
                           total_orders=5, total_spent=1750000,
                           first_order_date=today - timedelta(days=180), last_order_date=today - timedelta(days=5)),
            RepeatCustomer(customer_name="佐藤花子", email="sato@example.com", phone="090-3333-4444",
                           total_orders=12, total_spent=4200000,
                           first_order_date=today - timedelta(days=365), last_order_date=today - timedelta(days=12)),
            RepeatCustomer(customer_name="鈴木一郎", email="suzuki@example.com", phone="090-5555-6666",
                           total_orders=2, total_spent=65000,
                           first_order_date=today - timedelta(days=25), last_order_date=today - timedelta(days=20)),
        ]
        for c in customers:
            c.update_avg()
            c.segment = c.calc_segment()
        db.session.add_all(customers)
        print(f"  RepeatCustomer: {len(customers)}件")
    else:
        print("  RepeatCustomer: skip")

    # ListingProgress
    if not ListingProgress.query.first():
        cumulative = 0
        for i in range(7):
            d = today - timedelta(days=6 - i)
            count = random.randint(15, 25)
            cumulative += count
            db.session.add(ListingProgress(
                record_date=d, listings_count=count,
                target_daily=20, target_monthly=600,
                cumulative_monthly=cumulative,
            ))
        print("  ListingProgress: 7日分")
    else:
        print("  ListingProgress: skip")

    # StockCheck
    if not StockCheck.query.first():
        db.session.add_all([
            StockCheck(product_id=products[0].id, source_url="https://example.com/chanel-001",
                       current_price=440000, previous_price=450000, in_stock=True,
                       price_changed=True, checked_at=today - timedelta(hours=2)),
            StockCheck(product_id=products[1].id, source_url="https://example.com/hermes-001",
                       current_price=850000, previous_price=850000, in_stock=True,
                       price_changed=False, checked_at=today - timedelta(hours=4)),
            StockCheck(product_id=products[6].id, source_url="https://example.com/nike-001",
                       current_price=28000, previous_price=25000, in_stock=False,
                       price_changed=True, stock_changed=True, checked_at=today - timedelta(hours=6)),
        ])
        print("  StockCheck: 3件")
    else:
        print("  StockCheck: skip")

    # PopularityTracker
    if not PopularityTracker.query.first():
        for pid in [0, 2, 3, 4, 6]:
            pt = PopularityTracker(
                product_id=products[pid].id,
                views=random.randint(50, 500), favorites=random.randint(5, 50),
                inquiries=random.randint(0, 10), sold_count=random.randint(0, 5),
                tracking_date=today,
            )
            pt.popularity_score = pt.calc_score()
            db.session.add(pt)
        print("  PopularityTracker: 5件")
    else:
        print("  PopularityTracker: skip")

    # RegionRecommendation
    if not RegionRecommendation.query.first():
        for code, name, profit, ship, customs, risk, rel in [
            ("US", "アメリカ合衆国", 0.22, 7, 0.05, 20, 90),
            ("IT", "イタリア", 0.28, 5, 0.08, 15, 95),
            ("FR", "フランス", 0.25, 6, 0.07, 15, 95),
            ("UK", "イギリ国", 0.20, 5, 0.12, 25, 85),
            ("JP", "日本", 0.15, 2, 0.00, 5, 98),
        ]:
            rr = RegionRecommendation(
                region=code, region_name=name,
                avg_profit_rate=profit, avg_shipping_days=ship,
                avg_customs_rate=customs, risk_score=risk, reliability_score=rel,
                last_updated=today,
            )
            rr.recommendation_score = rr.calc_recommendation() or 0
            db.session.add(rr)
        print("  RegionRecommendation: 5件")
    else:
        print("  RegionRecommendation: skip")


def reset_all() -> None:
    """全テーブルのデータを削除"""
    tables = [
        PopularityTracker, StockCheck, ListingProgress,
        RegionRecommendation, RepeatCustomer, Order,
        ListingTemplate, Product, Partner, User,
    ]
    for model in tables:
        count = db.session.query(model).delete()
        if count:
            print(f"  Deleted {model.__tablename__}: {count}件")
    db.session.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed sample data (idempotent)")
    parser.add_argument("--reset", action="store_true", help="Delete all data before seeding")
    parser.add_argument("--admin-only", action="store_true", help="Create admin user only")
    parser.add_argument("--username", default="admin", help="Admin username")
    parser.add_argument("--password", default="changeme", help="Admin password")
    args = parser.parse_args()

    app = create_app()
    with app.app_context():
        if args.reset:
            print("=== Resetting all data ===")
            reset_all()

        print("=== Seeding sample data ===")
        seed_admin(username=args.username, password=args.password)

        if args.admin_only:
            db.session.commit()
            return

        products = seed_products()
        partners = seed_partners()
        seed_orders(products, partners)
        seed_templates()
        seed_remaining(products)
        db.session.commit()
        print("=== Done ===")


if __name__ == "__main__":
    main()
