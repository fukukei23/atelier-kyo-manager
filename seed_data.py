"""サンプルデータ投入スクリプト"""
from datetime import datetime, timedelta
import random

from app import create_app
from app.extensions import db
from app.models import (
    Product, Partner, Order, RepeatCustomer, ListingTemplate,
    ListingProgress, StockCheck, PopularityTracker, RegionRecommendation,
)

app = create_app()

with app.app_context():
    today = datetime.now()

    # ---- Product (10件) ----
    products = [
        Product(name="Classic Flap Bag", brand="Chanel", purchase_price=450000, selling_price=585000,
                supplier_url="https://example.com/product/chanel-001", image_url="https://placehold.co/300x300?text=Chanel",
                stock_status=True, source_type="overseas", source_region="FR", color="Black", size="M", material="Lambskin",
                target_profit_rate=0.30, listing_status="listed"),
        Product(name="Birkin 25", brand="Hermes", purchase_price=850000, selling_price=1105000,
                supplier_url="https://example.com/product/hermes-001", image_url="https://placehold.co/300x300?text=Hermes",
                stock_status=True, source_type="overseas", source_region="FR", color="Gold", size="25cm", material="Togo Leather",
                target_profit_rate=0.30, listing_status="listed"),
        Product(name="Neverfull MM", brand="Louis Vuitton", purchase_price=140000, selling_price=182000,
                supplier_url="https://example.com/product/lv-001", image_url="https://placehold.co/300x300?text=LV",
                stock_status=True, source_type="overseas", source_region="US", color="Damier Azur", size="MM", material="Canvas",
                target_profit_rate=0.25, listing_status="listed"),
        Product(name="GG Marmont Small", brand="Gucci", purchase_price=180000, selling_price=234000,
                supplier_url="https://example.com/product/gucci-001", image_url="https://placehold.co/300x300?text=Gucci",
                stock_status=True, source_type="overseas", source_region="IT", color="Gold", size="Small", material="Leather",
                target_profit_rate=0.25, listing_status="listed"),
        Product(name="Re-Edition 2005", brand="Prada", purchase_price=95000, selling_price=123500,
                supplier_url="https://example.com/product/prada-001", image_url="https://placehold.co/300x300?text=Prada",
                stock_status=True, source_type="overseas", source_region="IT", color="Black", size="One Size", material="Saffiano",
                target_profit_rate=0.25, listing_status="draft"),
        Product(name="Nylon Logo Jacket", brand="Moncler", purchase_price=120000, selling_price=156000,
                supplier_url="https://example.com/product/moncler-001", image_url="https://placehold.co/300x300?text=Moncler",
                stock_status=True, source_type="overseas", source_region="IT", color="Navy", size="M", material="Nylon",
                target_profit_rate=0.25, listing_status="draft"),
        Product(name="Air Jordan 1 Retro High", brand="Nike", purchase_price=25000, selling_price=32500,
                supplier_url="https://example.com/product/nike-001", image_url="https://placehold.co/300x300?text=Nike",
                stock_status=True, source_type="overseas", source_region="US", color="Chicago", size="27cm", material="Leather",
                target_profit_rate=0.25, listing_status="listed"),
        Product(name="Basic T-Shirt", brand="ZARA", purchase_price=2500, selling_price=3250,
                supplier_url="https://example.com/product/zara-001", image_url="https://placehold.co/300x300?text=ZARA",
                stock_status=True, source_type="domestic", source_region="ES", color="White", size="M", material="Cotton",
                target_profit_rate=0.25, listing_status="draft"),
        Product(name="Heattech Ultra Warm", brand="UNIQLO", purchase_price=1500, selling_price=1950,
                supplier_url="https://example.com/product/uniqlo-001", image_url="https://placehold.co/300x300?text=UNIQLO",
                stock_status=False, source_type="domestic", source_region="JP", color="Black", size="L", material="Polyester",
                target_profit_rate=0.25, listing_status="sold"),
        Product(name="Cotton Blazer", brand="H&M", purchase_price=4500, selling_price=5850,
                supplier_url="https://example.com/product/hm-001", image_url="https://placehold.co/300x300?text=HM",
                stock_status=True, source_type="domestic", source_region="SE", color="Gray", size="M", material="Cotton",
                target_profit_rate=0.25, listing_status="archived"),
    ]
    for p in products:
        p.auto_classify_tier()
    db.session.add_all(products)
    db.session.flush()  # IDs確保

    # ---- Partner (3件) ----
    partners = [
        Partner(name="Global Fashion Hub", email="contact@globalfashion.com", phone="+1-212-555-0100",
                active_regions="US,CA", specialty_brands="Chanel,Louis Vuitton,Gucci",
                priority_level="high", status="active"),
        Partner(name="Euro Luxe Partners", email="info@euroluxe.eu", phone="+39-02-555-0200",
                active_regions="IT,FR,DE,UK", specialty_brands="Hermes,Prada,Moncler",
                priority_level="medium", status="active"),
        Partner(name="Tokyo Connect", email="sales@tokyoconnect.jp", phone="03-5555-0300",
                active_regions="JP", specialty_brands="UNIQLO,ZARA",
                priority_level="low", status="active"),
    ]
    db.session.add_all(partners)
    db.session.flush()

    # ---- Order (5件) ----
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

    # ---- RepeatCustomer (3件) ----
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

    # ---- ListingTemplate (2件) ----
    db.session.add_all([
        ListingTemplate(name="標準テンプレート",
                        template_text="【{{brand}}】{{productName}}\n\n★ BUYMA最安値挑戦！★\n\n✅正規品保証 ✅最短翌日発送 ✅関税込み\n\n📦カラー: {{color}}\n📏サイズ: {{size}}\n🧵素材: {{material}}\n\n即決歓迎！お気軽にご質問ください♪",
                        category="general", is_default=True),
        ListingTemplate(name="ファッション用テンプレート",
                        template_text="🛍️ {{brand}} {{productName}} 🛍️\n\n✨ブランド正規品✨\n\nブランド: {{brand}}\nカラー: {{color}}\nサイズ: {{size}}\n素材: {{material}}\n\n即決フォロー割あり！",
                        category="fashion", is_default=False),
    ])

    # ---- ListingProgress (7日分) ----
    cumulative = 0
    for i in range(7):
        d = today - timedelta(days=6 - i)
        count = random.randint(15, 25)
        cumulative += count
        db.session.add(ListingProgress(
            record_date=d, listings_count=count,
            target_daily=20, target_monthly=600,
            cumulative_monthly=cumulative, notes="",
        ))

    # ---- StockCheck (3件) ----
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

    # ---- PopularityTracker (5件) ----
    for pid in [0, 2, 3, 4, 6]:
        pt = PopularityTracker(
            product_id=products[pid].id,
            views=random.randint(50, 500), favorites=random.randint(5, 50),
            inquiries=random.randint(0, 10), sold_count=random.randint(0, 5),
            tracking_date=today,
        )
        pt.popularity_score = pt.calc_score()
        db.session.add(pt)

    # ---- RegionRecommendation (5件) ----
    for code, name, profit, ship, customs, risk, rel in [
        ("US", "アメリカ合衆国", 0.22, 7, 0.05, 20, 90),
        ("IT", "イタリア", 0.28, 5, 0.08, 15, 95),
        ("FR", "フランス", 0.25, 6, 0.07, 15, 95),
        ("UK", "イギリス", 0.20, 5, 0.12, 25, 85),
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

    db.session.commit()

    # 結果サマリー
    print(f"=== サンプルデータ投入完了 ===")
    print(f"  Product:           {len(products)}件")
    print(f"  Partner:           {len(partners)}件")
    print(f"  Order:             {len(orders)}件")
    print(f"  RepeatCustomer:    {len(customers)}件")
    print(f"  ListingTemplate:   2件")
    print(f"  ListingProgress:   7日分")
    print(f"  StockCheck:        3件")
    print(f"  PopularityTracker: 5件")
    print(f"  RegionRecommendation: 5件")
