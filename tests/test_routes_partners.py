"""ルートテスト: Partners（CRUD）+ Customers（repeat customer CRUD）"""

from __future__ import annotations

import pytest

from app.extensions import db


class TestPartnerRoutes:
    def test_partner_list_200(self, routes_auth_client):
        r = routes_auth_client.get("/partners")
        assert r.status_code == 200

    def test_partner_new_form_200(self, routes_auth_client):
        r = routes_auth_client.get("/partners/new")
        assert r.status_code == 200

    def test_partner_create_submit(self, routes_auth_client, routes_app):
        r = routes_auth_client.post(
            "/partners/new",
            data={
                "name": "TestPartner",
                "email": "p@example.com",
                "phone": "090-1234-5678",
                "active_regions": "EU,US",
                "specialty_brands": "Gucci,Prada",
                "priority_level": "high",
                "status": "active",
                "notes": "test partner",
            },
            follow_redirects=False,
        )
        assert r.status_code == 302
        with routes_app.app_context():
            from app.models.partner import Partner
            p = Partner.query.filter_by(name="TestPartner").first()
            assert p is not None
            assert p.email == "p@example.com"

    def test_partner_create_missing_name(self, routes_auth_client):
        r = routes_auth_client.post("/partners/new", data={"name": ""})
        assert r.status_code == 200

    def test_partner_edit_200(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            from app.models.partner import Partner
            p = Partner(name="EditTarget", email="old@example.com")
            db.session.add(p)
            db.session.commit()
            pid = p.id
        r = routes_auth_client.get(f"/partners/{pid}/edit")
        assert r.status_code == 200

    def test_partner_edit_submit(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            from app.models.partner import Partner
            p = Partner(name="BeforeEdit")
            db.session.add(p)
            db.session.commit()
            pid = p.id
        r = routes_auth_client.post(
            f"/partners/{pid}/edit",
            data={"name": "AfterEdit", "email": "new@example.com"},
            follow_redirects=False,
        )
        assert r.status_code == 302
        with routes_app.app_context():
            from app.models.partner import Partner
            updated = db.session.get(Partner, pid)
            assert updated.name == "AfterEdit"

    def test_partner_edit_missing_name(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            from app.models.partner import Partner
            p = Partner(name="NoNameEdit")
            db.session.add(p)
            db.session.commit()
            pid = p.id
        r = routes_auth_client.post(f"/partners/{pid}/edit", data={"name": ""})
        assert r.status_code == 200

    def test_partner_delete(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            from app.models.partner import Partner
            p = Partner(name="DeleteTarget")
            db.session.add(p)
            db.session.commit()
            pid = p.id
        r = routes_auth_client.post(f"/partners/{pid}/delete", follow_redirects=True)
        assert r.status_code == 200
        with routes_app.app_context():
            from app.models.partner import Partner
            assert db.session.get(Partner, pid) is None


class TestCustomerRoutes:
    def test_customer_list_200(self, routes_auth_client):
        r = routes_auth_client.get("/customers")
        assert r.status_code == 200

    def test_customer_new_form_200(self, routes_auth_client):
        r = routes_auth_client.get("/customers/new")
        assert r.status_code == 200

    def test_customer_create_submit(self, routes_auth_client, routes_app):
        r = routes_auth_client.post(
            "/customers/new",
            data={
                "customer_name": "TestCustomer",
                "email": "c@example.com",
                "total_orders": "3",
                "total_spent": "15000",
                "notes": "test",
            },
            follow_redirects=True,
        )
        assert r.status_code == 200
        with routes_app.app_context():
            from app.models.repeat_customer import RepeatCustomer
            c = RepeatCustomer.query.filter_by(customer_name="TestCustomer").first()
            assert c is not None
            assert c.total_orders == 3

    def test_customer_create_negative_orders(self, routes_auth_client):
        r = routes_auth_client.post(
            "/customers/new",
            data={"customer_name": "Bad", "total_orders": "-1", "total_spent": "0"},
        )
        assert r.status_code == 200

    def test_customer_create_invalid_date(self, routes_auth_client):
        r = routes_auth_client.post(
            "/customers/new",
            data={
                "customer_name": "BadDate",
                "total_orders": "1",
                "first_order_date": "not-a-date",
            },
        )
        assert r.status_code == 200

    def test_customer_edit_200(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            from app.models.repeat_customer import RepeatCustomer
            c = RepeatCustomer(customer_name="EditCust", total_orders=1, total_spent=1000)
            c.segment = c.calc_segment()
            db.session.add(c)
            db.session.commit()
            cid = c.id
        r = routes_auth_client.get(f"/customers/{cid}/edit")
        assert r.status_code == 200

    def test_customer_edit_submit(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            from app.models.repeat_customer import RepeatCustomer
            c = RepeatCustomer(customer_name="EditCust2", total_orders=1, total_spent=1000)
            c.segment = c.calc_segment()
            db.session.add(c)
            db.session.commit()
            cid = c.id
        r = routes_auth_client.post(
            f"/customers/{cid}/edit",
            data={"customer_name": "UpdatedCust", "total_orders": "5", "total_spent": "5000"},
            follow_redirects=True,
        )
        assert r.status_code == 200

    def test_customer_delete(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            from app.models.repeat_customer import RepeatCustomer
            c = RepeatCustomer(customer_name="DeleteCust", total_orders=0, total_spent=0)
            db.session.add(c)
            db.session.commit()
            cid = c.id
        r = routes_auth_client.post(f"/customers/{cid}/delete", follow_redirects=True)
        assert r.status_code == 200
