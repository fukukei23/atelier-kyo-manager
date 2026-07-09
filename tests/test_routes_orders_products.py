"""ルートテスト: Auto Orders + Products（CRUD・CSV・パイプライン）"""

from __future__ import annotations

from app.extensions import db


class TestAutoOrderRoutes:
    def test_auto_orders_list_200(self, routes_auth_client):
        r = routes_auth_client.get("/auto-orders")
        assert r.status_code == 200

    def test_auto_orders_requires_auth(self, routes_client):
        r = routes_client.get("/auto-orders", follow_redirects=False)
        assert r.status_code == 302

    def test_start_auto_order_404(self, routes_auth_client):
        """handle_db_error が NotFound を捕捉して 302 リダイレクトすることを確認。"""
        r = routes_auth_client.post("/auto-orders/9999/start", follow_redirects=False)
        assert r.status_code == 302

    def test_execute_auto_order_step_404(self, routes_auth_client):
        r = routes_auth_client.post("/auto-orders/9999/step", follow_redirects=False)
        assert r.status_code == 302

    def test_report_auto_order_error_404(self, routes_auth_client):
        r = routes_auth_client.post("/auto-orders/9999/error", follow_redirects=False)
        assert r.status_code == 302

    def test_auto_orders_start_with_order(self, routes_auth_client, routes_app):
        """実際の Order を使った start のテスト。"""
        with routes_app.app_context():
            from app.models.order import Order

            o = Order(order_number="ORD-001", product_name="Test", status="pending")
            db.session.add(o)
            db.session.commit()
            oid = o.id
        r = routes_auth_client.post(f"/auto-orders/{oid}/start", follow_redirects=True)
        assert r.status_code == 200


class TestProductRoutes:
    def test_index_200(self, routes_auth_client):
        r = routes_auth_client.get("/")
        assert r.status_code == 200

    def test_index_no_auth(self, routes_client):
        r = routes_client.get("/")
        assert r.status_code == 200  # index は公開

    def test_manage_products_200(self, routes_auth_client):
        r = routes_auth_client.get("/manage")
        assert r.status_code == 200

    def test_product_list_200(self, routes_auth_client):
        r = routes_auth_client.get("/products")
        assert r.status_code == 200

    def test_product_edit_404(self, routes_auth_client):
        r = routes_auth_client.get("/products/9999/edit", follow_redirects=False)
        assert r.status_code == 302

    def test_product_delete_404(self, routes_auth_client):
        r = routes_auth_client.post("/products/9999/delete", follow_redirects=False)
        assert r.status_code == 302

    def test_export_csv_200(self, routes_auth_client):
        r = routes_auth_client.get("/export_csv")
        assert r.status_code == 200
        assert "text/csv" in r.content_type

    def test_listing_candidates_200(self, routes_auth_client):
        r = routes_auth_client.get("/listing-candidates")
        assert r.status_code == 200

    def test_listing_candidates_export_200(self, routes_auth_client):
        r = routes_auth_client.get("/listing-candidates/export")
        assert r.status_code == 200
        assert "text/csv" in r.content_type

    def test_import_csv_no_file(self, routes_auth_client):
        r = routes_auth_client.post("/import_csv", follow_redirects=True)
        assert r.status_code == 200

    def test_import_csv_non_csv_file(self, routes_auth_client):
        import io

        data = {"file": (io.BytesIO(b"not a csv"), "test.txt")}
        r = routes_auth_client.post("/import_csv", data=data, follow_redirects=True)
        assert r.status_code == 200

    def test_form_redirect(self, routes_client):
        r = routes_client.get("/form", follow_redirects=False)
        assert r.status_code == 302

    def test_list_redirect(self, routes_client):
        r = routes_client.get("/list", follow_redirects=False)
        assert r.status_code == 302

    def test_generate_listing_404(self, routes_auth_client):
        r = routes_auth_client.get("/products/9999/generate-listing")
        assert r.status_code == 404

    def test_pipeline_result_404(self, routes_auth_client):
        r = routes_auth_client.get("/products/9999/pipeline-result")
        assert r.status_code == 404

    def test_upload_images_404(self, routes_auth_client):
        r = routes_auth_client.post("/products/9999/upload-images", follow_redirects=True)
        assert r.status_code == 404

    def test_run_pipeline_404(self, routes_auth_client):
        r = routes_auth_client.post("/products/9999/run-pipeline", follow_redirects=True)
        assert r.status_code == 404

    def test_run_pipeline_batch_no_ids(self, routes_auth_client):
        r = routes_auth_client.post("/run-pipeline-batch", follow_redirects=True)
        assert r.status_code == 200
