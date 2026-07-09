from __future__ import annotations

from datetime import date as _date

from app.extensions import db
from app.models.popularity_tracker import PopularityTracker
from app.models.product import Product

# ---------------------------------------------------------------------------
# Model unit tests
# ---------------------------------------------------------------------------


class TestPopularityTrackerModel:
    def test_calc_score_normal_values(self):
        tracker = PopularityTracker(views=100, favorites=10, inquiries=5, sold_count=2)
        expected = 100 * 0.1 + 10 * 2 + 5 * 3 + 2 * 5
        assert tracker.calc_score() == expected

    def test_calc_score_zero_values(self):
        tracker = PopularityTracker(views=0, favorites=0, inquiries=0, sold_count=0)
        assert tracker.calc_score() == 0.0

    def test_calc_score_large_values(self):
        tracker = PopularityTracker(views=10000, favorites=1000, inquiries=500, sold_count=100)
        expected = 10000 * 0.1 + 1000 * 2 + 500 * 3 + 100 * 5
        assert tracker.calc_score() == expected

    def test_calc_score_none_fields(self):
        tracker = PopularityTracker(views=None, favorites=None, inquiries=None, sold_count=None)
        assert tracker.calc_score() == 0.0

    def test_sold_count_or_zero_with_value(self):
        tracker = PopularityTracker(sold_count=5)
        assert tracker.sold_count_or_zero() == 5

    def test_sold_count_or_zero_with_none(self):
        tracker = PopularityTracker(sold_count=None)
        assert tracker.sold_count_or_zero() == 0

    def test_sold_count_or_zero_with_zero(self):
        tracker = PopularityTracker(sold_count=0)
        assert tracker.sold_count_or_zero() == 0


# ---------------------------------------------------------------------------
# Route tests
# ---------------------------------------------------------------------------


def _create_product(app) -> int:
    with app.app_context():
        p = Product(name="テスト商品", brand="Brand", purchase_price=1000.0, selling_price=1500.0)
        db.session.add(p)
        db.session.commit()
        return p.id


class TestPopularityRoutes:
    def test_get_list_unauthenticated(self, routes_client):
        resp = routes_client.get("/popularity")
        assert resp.status_code in (200, 302)

    def test_get_list_authenticated(self, routes_auth_client):
        resp = routes_auth_client.get("/popularity")
        assert resp.status_code == 200

    def test_get_new_form_authenticated(self, routes_auth_client):
        resp = routes_auth_client.get("/popularity/new")
        assert resp.status_code == 200

    def test_post_new_valid_data(self, routes_auth_client, routes_app):
        product_id = _create_product(routes_app)
        resp = routes_auth_client.post(
            "/popularity/new",
            data={
                "product_id": str(product_id),
                "views": "100",
                "favorites": "10",
                "inquiries": "5",
                "sold_count": "2",
            },
            follow_redirects=False,
        )
        assert resp.status_code == 302

    def test_post_new_invalid_string(self, routes_auth_client, routes_app):
        product_id = _create_product(routes_app)
        resp = routes_auth_client.post(
            "/popularity/new",
            data={
                "product_id": str(product_id),
                "views": "invalid",
                "favorites": "10",
                "inquiries": "5",
                "sold_count": "2",
            },
        )
        assert resp.status_code == 200

    def test_post_new_negative_values(self, routes_auth_client, routes_app):
        product_id = _create_product(routes_app)
        resp = routes_auth_client.post(
            "/popularity/new",
            data={
                "product_id": str(product_id),
                "views": "-10",
                "favorites": "10",
                "inquiries": "5",
                "sold_count": "2",
            },
        )
        assert resp.status_code == 200

    def test_post_new_all_zero(self, routes_auth_client, routes_app):
        product_id = _create_product(routes_app)
        resp = routes_auth_client.post(
            "/popularity/new",
            data={
                "product_id": str(product_id),
                "views": "0",
                "favorites": "0",
                "inquiries": "0",
                "sold_count": "0",
            },
            follow_redirects=False,
        )
        assert resp.status_code == 302

    def test_delete_existing(self, routes_auth_client, routes_app):
        product_id = _create_product(routes_app)
        with routes_app.app_context():
            pt = PopularityTracker(
                product_id=product_id,
                views=10,
                favorites=2,
                inquiries=1,
                sold_count=1,
                tracking_date=_date.today(),
            )
            pt.popularity_score = pt.calc_score()
            db.session.add(pt)
            db.session.commit()
            tid = pt.id
        resp = routes_auth_client.post(f"/popularity/{tid}/delete", follow_redirects=False)
        assert resp.status_code in (302, 404)

    def test_delete_nonexistent(self, routes_auth_client):
        resp = routes_auth_client.post("/popularity/99999/delete", follow_redirects=False)
        assert resp.status_code in (302, 404)

    def test_get_list_pagination(self, routes_auth_client):
        resp = routes_auth_client.get("/popularity?page=2")
        assert resp.status_code == 200
