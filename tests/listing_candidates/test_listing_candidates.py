"""FR-007 出品候補リスト テスト"""
import pytest

from app import create_app
from app.extensions import db
from app.models import Product
from app.models.user import User


@pytest.fixture(scope="function")
def app():
    _app = create_app()
    _app.config.update({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
        "WTF_CSRF_ENABLED": False,
        "SECRET_KEY": "test-secret",
    })
    with _app.app_context():
        db.create_all()
        yield _app
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def test_user(app):
    with app.app_context():
        u = User(username="testuser", display_name="Test", is_active=True)
        u.set_password("password123")
        db.session.add(u)
        db.session.commit()
    return {"username": "testuser", "password": "password123"}


@pytest.fixture()
def auth_client(client, test_user):
    client.post("/auth/login", data=test_user, follow_redirects=False)
    return client


# ---- listing_candidates() モデルテスト ----

def test_listing_candidates_filters_listed(app):
    """listing_status='listed' は除外される"""
    with app.app_context():
        # purchase=1000, selling=1500 → rate=4.3% (手数料引後)
        # min_profit_rate=0 で全件取得
        p1 = Product(
            name="候補A", purchase_price=1000, selling_price=1500,
            stock_status=True, listing_status="draft",
        )
        p2 = Product(
            name="出品済", purchase_price=1000, selling_price=1500,
            stock_status=True, listing_status="listed",
        )
        db.session.add_all([p1, p2])
        db.session.commit()

        candidates = Product.listing_candidates(min_profit_rate=0)
        names = [c.name for c in candidates]
        assert "候補A" in names
        assert "出品済" not in names


def test_listing_candidates_filters_no_stock(app):
    """stock_status=False は除外される"""
    with app.app_context():
        p1 = Product(
            name="在庫あり", purchase_price=1000, selling_price=1500,
            stock_status=True, listing_status="draft",
        )
        p2 = Product(
            name="在庫なし", purchase_price=1000, selling_price=1500,
            stock_status=False, listing_status="draft",
        )
        db.session.add_all([p1, p2])
        db.session.commit()

        candidates = Product.listing_candidates(min_profit_rate=0)
        names = [c.name for c in candidates]
        assert "在庫あり" in names
        assert "在庫なし" not in names


def test_listing_candidates_filters_by_profit_rate(app):
    """利益率フィルタが機能する"""
    with app.app_context():
        # purchase=1000, selling=2000 → rate=26.3%
        p1 = Product(
            name="高利益", purchase_price=1000, selling_price=2000,
            stock_status=True, listing_status="draft",
        )
        # purchase=1000, selling=1050 → rate=マイナス
        p2 = Product(
            name="低利益", purchase_price=1000, selling_price=1050,
            stock_status=True, listing_status="draft",
        )
        db.session.add_all([p1, p2])
        db.session.commit()

        # min 20% → 高利益のみ
        candidates = Product.listing_candidates(min_profit_rate=20.0)
        names = [c.name for c in candidates]
        assert "高利益" in names
        assert "低利益" not in names


def test_listing_candidates_sort_by_tier(app):
    """brand_tier順 (high→medium→low) でソートされる"""
    with app.app_context():
        p_low = Product(
            name="ロー", purchase_price=1000, selling_price=2000,
            stock_status=True, listing_status="draft", brand_tier="low",
        )
        p_high = Product(
            name="ハイ", purchase_price=1000, selling_price=2000,
            stock_status=True, listing_status="draft", brand_tier="high",
        )
        p_med = Product(
            name="ミドル", purchase_price=1000, selling_price=2000,
            stock_status=True, listing_status="draft", brand_tier="medium",
        )
        db.session.add_all([p_low, p_high, p_med])
        db.session.commit()

        candidates = Product.listing_candidates(min_profit_rate=0)
        tiers = [c.brand_tier for c in candidates]
        assert tiers == ["high", "medium", "low"]


def test_listing_candidates_default_min_rate(app):
    """デフォルト最低利益率(10%)が適用される"""
    with app.app_context():
        # purchase=1000, selling=1500 → rate=4.3% < 10%
        p1 = Product(
            name="低利益", purchase_price=1000, selling_price=1500,
            stock_status=True, listing_status="draft",
        )
        db.session.add(p1)
        db.session.commit()

        candidates = Product.listing_candidates()  # デフォルト10%
        assert len(candidates) == 0


def test_listing_candidates_empty_db(app):
    """商品なしの場合は空リストを返す"""
    with app.app_context():
        candidates = Product.listing_candidates()
        assert candidates == []


# ---- ルートテスト（認証済み） ----

def test_listing_candidates_route(auth_client, app):
    """GET /listing-candidates が200を返す"""
    with app.app_context():
        p = Product(
            name="候補商品", purchase_price=1000, selling_price=2000,
            stock_status=True, listing_status="draft",
        )
        db.session.add(p)
        db.session.commit()

    resp = auth_client.get("/listing-candidates")
    assert resp.status_code == 200


def test_export_listing_candidates_csv(auth_client, app):
    """GET /listing-candidates/export がCSVを返す"""
    with app.app_context():
        p = Product(
            name="CSV候補", purchase_price=1000, selling_price=2000,
            stock_status=True, listing_status="draft",
        )
        db.session.add(p)
        db.session.commit()

    resp = auth_client.get("/listing-candidates/export")
    assert resp.status_code == 200
    assert resp.content_type.startswith("text/csv")
    assert "CSV候補".encode() in resp.data


def test_listing_candidates_route_with_filter(auth_client, app):
    """min_profit_rate パラメータでフィルタが動作する"""
    with app.app_context():
        p = Product(
            name="高利益", purchase_price=1000, selling_price=2000,
            stock_status=True, listing_status="draft",
        )
        db.session.add(p)
        db.session.commit()

    # 高い閾値 → 0件
    resp = auth_client.get("/listing-candidates?min_profit_rate=90")
    assert resp.status_code == 200
    assert "出品候補が見つかりません".encode() in resp.data
