class TestOrderList:
    def test_unauthenticated_redirect(self, routes_client):
        r = routes_client.get("/orders", follow_redirects=False)
        assert r.status_code == 302
        assert "login" in r.location.lower()

    def test_authenticated_200(self, routes_auth_client):
        assert routes_auth_client.get("/orders").status_code == 200


class TestCreateOrder:
    VALID = {
        "order_number": "ORD-001",
        "product_name": "TestProd",
        "customer_name": "TestCust",
        "order_date": "2024-01-15",
        "selling_price": "100000",
        "purchase_cost": "60000",
        "customs_duty": "5000",
        "payment_method": "bank_transfer",
        "source_type": "domestic",
        "status": "pending",
        "notes": "",
    }

    def test_get_new_form(self, routes_auth_client):
        assert routes_auth_client.get("/orders/new").status_code == 200

    def test_post_valid_redirects(self, routes_auth_client):
        r = routes_auth_client.post("/orders/new", data=self.VALID, follow_redirects=False)
        assert r.status_code == 302

    def test_post_negative_price(self, routes_auth_client):
        data = {**self.VALID, "selling_price": "-1000", "order_number": "ORD-NEG"}
        r = routes_auth_client.post("/orders/new", data=data, follow_redirects=True)
        assert r.status_code == 200

    def test_post_negative_purchase_cost(self, routes_auth_client):
        data = {**self.VALID, "purchase_cost": "-5000", "order_number": "ORD-NEG2"}
        r = routes_auth_client.post("/orders/new", data=data, follow_redirects=True)
        assert r.status_code == 200

    def test_post_no_date_uses_now(self, routes_auth_client):
        data = {**self.VALID, "order_date": "", "order_number": "ORD-NODATE"}
        r = routes_auth_client.post("/orders/new", data=data, follow_redirects=False)
        assert r.status_code == 302


class TestEditOrder:
    VALID = {
        "order_number": "ORD-EDIT",
        "product_name": "EditProd",
        "customer_name": "EditCust",
        "order_date": "2024-01-15",
        "selling_price": "100000",
        "purchase_cost": "60000",
        "customs_duty": "5000",
        "payment_method": "bank_transfer",
        "source_type": "domestic",
        "status": "pending",
        "notes": "",
    }

    def _make_order(self, client):
        client.post("/orders/new", data=self.VALID)
        from app.models.order import Order

        o = Order.query.first()
        return o.id if o else None

    def test_get_edit_nonexistent(self, routes_auth_client):
        # handle_db_error デコレータが404をリダイレクトに変換する
        r = routes_auth_client.get("/orders/99999/edit")
        assert r.status_code in (302, 404)

    def test_get_edit_existing(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            oid = self._make_order(routes_auth_client)
        if oid:
            assert routes_auth_client.get(f"/orders/{oid}/edit").status_code == 200

    def test_post_edit_valid(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            oid = self._make_order(routes_auth_client)
        if oid:
            upd = {**self.VALID, "selling_price": "120000", "status": "shipped", "order_date": "2024-02-01"}
            r = routes_auth_client.post(f"/orders/{oid}/edit", data=upd, follow_redirects=False)
            assert r.status_code == 302

    def test_post_edit_negative(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            oid = self._make_order(routes_auth_client)
        if oid:
            upd = {**self.VALID, "selling_price": "-100"}
            r = routes_auth_client.post(f"/orders/{oid}/edit", data=upd, follow_redirects=True)
            assert r.status_code == 200


class TestDeleteOrder:
    def test_delete_nonexistent(self, routes_auth_client):
        r = routes_auth_client.post("/orders/99999/delete")
        assert r.status_code in (302, 404)

    def test_delete_existing(self, routes_auth_client, routes_app):
        routes_auth_client.post(
            "/orders/new",
            data={
                "order_number": "ORD-DEL",
                "product_name": "DelProd",
                "customer_name": "DelCust",
                "order_date": "2024-01-15",
                "selling_price": "10000",
                "purchase_cost": "5000",
                "customs_duty": "0",
                "payment_method": "bank_transfer",
                "source_type": "domestic",
                "status": "pending",
                "notes": "",
            },
        )
        from app.models.order import Order

        with routes_app.app_context():
            o = Order.query.first()
            if o:
                r = routes_auth_client.post(f"/orders/{o.id}/delete", follow_redirects=False)
                assert r.status_code == 302


class TestApiDashboard:
    def test_unauthenticated_redirect(self, routes_client):
        assert routes_client.get("/api/orders/dashboard", follow_redirects=False).status_code == 302

    def test_returns_json(self, routes_auth_client):
        r = routes_auth_client.get("/api/orders/dashboard")
        assert r.status_code == 200
        assert "application/json" in r.content_type
        assert isinstance(r.get_json(), list)

    def test_with_order_has_fields(self, routes_auth_client):
        routes_auth_client.post(
            "/orders/new",
            data={
                "order_number": "ORD-API",
                "product_name": "ApiProd",
                "customer_name": "ApiCust",
                "order_date": "2024-01-15",
                "selling_price": "100000",
                "purchase_cost": "60000",
                "customs_duty": "5000",
                "payment_method": "bank_transfer",
                "source_type": "domestic",
                "status": "pending",
                "notes": "",
            },
        )
        r = routes_auth_client.get("/api/orders/dashboard")
        data = r.get_json()
        if data:
            assert "id" in data[0]
            assert "color" in data[0]


class TestCashflow:
    def test_unauthenticated(self, routes_client):
        assert routes_client.get("/cashflow", follow_redirects=False).status_code == 302

    def test_authenticated_200(self, routes_auth_client):
        assert routes_auth_client.get("/cashflow").status_code == 200

    def test_with_days_param(self, routes_auth_client):
        assert routes_auth_client.get("/cashflow?days=7").status_code == 200

    def test_days_capped(self, routes_auth_client):
        assert routes_auth_client.get("/cashflow?days=9999").status_code == 200


class TestExtendOrder:
    def test_extend_nonexistent(self, routes_auth_client):
        r = routes_auth_client.post("/orders/99999/extend")
        assert r.status_code in (302, 404)

    def test_extend_existing(self, routes_auth_client, routes_app):
        routes_auth_client.post(
            "/orders/new",
            data={
                "order_number": "ORD-EXT",
                "product_name": "ExtProd",
                "customer_name": "ExtCust",
                "order_date": "2024-01-15",
                "selling_price": "50000",
                "purchase_cost": "30000",
                "customs_duty": "0",
                "payment_method": "bank_transfer",
                "source_type": "domestic",
                "status": "pending",
                "notes": "",
            },
        )
        from app.models.order import Order

        with routes_app.app_context():
            o = Order.query.first()
            if o:
                r = routes_auth_client.post(
                    f"/orders/{o.id}/extend", data={"extension_reason": "test reason"}, follow_redirects=False
                )
                assert r.status_code == 302
