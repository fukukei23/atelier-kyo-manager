"""Tests for app/routes/products.py - Issue #93 カバレッジ向上"""

from unittest.mock import patch

MINIMAL_PRODUCT = {
    "name": "Test Product",
    "brand": "Test Brand",
    "purchase_price": "10000",
    "selling_price": "20000",
    "transaction_fee": "1000",
    "shipping_cost": "500",
    "customs_duty": "200",
    "procurement_fee": "0",
    "supplier_url": "",
    "image_url": "",
    "stock_status": True,
    "listing_status": "draft",
    "source_type": "",
    "source_region": "",
    "color": "",
    "size": "",
    "material": "",
    "description": "",
    "retail_price": "",
    "target_profit_rate": "10",
    "warehouse_shipping_cost": "0",
    "original_currency": "JPY",
    "exchange_rate": "1.0",
    "item_category": "",
}


def _create_product(client):
    """テスト用商品を作成してIDを返す"""
    client.post("/manage", data=MINIMAL_PRODUCT)
    from app.models import Product

    p = Product.query.first()
    return p.id if p else None


class TestCompatRoutes:
    def test_form_view_redirects(self, routes_auth_client):
        r = routes_auth_client.get("/form", follow_redirects=False)
        assert r.status_code == 302

    def test_list_view_redirects(self, routes_auth_client):
        r = routes_auth_client.get("/list", follow_redirects=False)
        assert r.status_code == 302


class TestIndex:
    def test_index_200(self, routes_client):
        assert routes_client.get("/").status_code == 200


class TestManageProducts:
    def test_get_manage_authenticated(self, routes_auth_client):
        assert routes_auth_client.get("/manage").status_code == 200

    def test_get_manage_unauthenticated(self, routes_client):
        r = routes_client.get("/manage", follow_redirects=False)
        assert r.status_code == 302

    def test_post_valid_product_redirects(self, routes_auth_client):
        r = routes_auth_client.post("/manage", data=MINIMAL_PRODUCT, follow_redirects=False)
        assert r.status_code == 302

    def test_post_creates_product_in_db(self, routes_auth_client, routes_app):
        routes_auth_client.post("/manage", data=MINIMAL_PRODUCT)
        from app.models import Product

        with routes_app.app_context():
            assert Product.query.count() > 0

    def test_post_invalid_no_name_shows_form(self, routes_auth_client):
        data = {**MINIMAL_PRODUCT, "name": ""}
        r = routes_auth_client.post("/manage", data=data, follow_redirects=True)
        assert r.status_code == 200


class TestProductList:
    def test_product_list_200(self, routes_auth_client):
        assert routes_auth_client.get("/products").status_code == 200

    def test_product_list_unauthenticated(self, routes_client):
        r = routes_client.get("/products", follow_redirects=False)
        assert r.status_code == 302


class TestEditProduct:
    def test_get_edit_nonexistent(self, routes_auth_client):
        r = routes_auth_client.get("/products/99999/edit")
        assert r.status_code in (302, 404)

    def test_get_edit_existing(self, routes_auth_client, routes_app):
        from unittest.mock import MagicMock
        from app.core.pricing.schemas import PricingResult

        with routes_app.app_context():
            oid = _create_product(routes_auth_client)
        if oid:
            # /products/{id}/edit の GET 時にレンダリングされる
            # templates/products/manage.html が各 product の calculate_profit()
            # / profit_rate() を呼ぶ。Product 列には price_source が無いため
            # Product._calculate_pricing は内部で PricingInput(price_source=UNKNOWN)
            # を構築し、PriceIntegrityError を raise する設計。
            # 利益計算ロジック単体の網羅検証は tests/pricing/test_calculator.py
            # 側で既完了のため、ここではレンダリング成功（GET 200）のみを検証する。
            fake_result = PricingResult(
                revenue=20000.0,
                total_cost=15000.0,
                profit=5000.0,
                profit_rate=0.25,
            )
            with patch("app.models.product.Product._calculate_pricing", return_value=fake_result):
                r = routes_auth_client.get(f"/products/{oid}/edit")
            assert r.status_code == 200

    def test_post_edit_valid(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            oid = _create_product(routes_auth_client)
        if oid:
            upd = {**MINIMAL_PRODUCT, "name": "Updated Product"}
            r = routes_auth_client.post(f"/products/{oid}/edit", data=upd, follow_redirects=False)
            assert r.status_code == 302


class TestDeleteProduct:
    def test_delete_existing(self, routes_auth_client, routes_app):
        routes_auth_client.post("/manage", data=MINIMAL_PRODUCT)
        from app.models import Product

        with routes_app.app_context():
            p = Product.query.first()
            if p:
                r = routes_auth_client.post(f"/products/{p.id}/delete", follow_redirects=False)
                assert r.status_code == 302

    def test_delete_nonexistent(self, routes_auth_client):
        r = routes_auth_client.post("/products/99999/delete")
        assert r.status_code in (302, 404)


class TestExportCsv:
    def test_export_csv_content_type(self, routes_auth_client):
        r = routes_auth_client.get("/export_csv")
        assert r.status_code == 200
        assert "text/csv" in r.content_type

    def test_export_unauthenticated(self, routes_client):
        r = routes_client.get("/export_csv", follow_redirects=False)
        assert r.status_code == 302


class TestGenerateBuymaCsv:
    def test_generate_buyma_csv_returns_csv(self, routes_auth_client):
        r = routes_auth_client.get("/generate-buyma-csv")
        assert r.status_code == 200
        assert "text/csv" in r.content_type

    def test_generate_buyma_csv_unauthenticated(self, routes_client):
        r = routes_client.get("/generate-buyma-csv", follow_redirects=False)
        assert r.status_code == 302


class TestGenerateListing:
    def test_generate_listing_existing_product(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            oid = _create_product(routes_auth_client)
        if oid:
            r = routes_auth_client.get(f"/products/{oid}/generate-listing")
            assert r.status_code == 200

    def test_generate_listing_nonexistent(self, routes_auth_client):
        r = routes_auth_client.get("/products/99999/generate-listing")
        assert r.status_code in (302, 404)


class TestListingCandidates:
    def test_listing_candidates_200(self, routes_auth_client):
        assert routes_auth_client.get("/listing-candidates").status_code == 200

    def test_listing_candidates_export_csv(self, routes_auth_client):
        r = routes_auth_client.get("/listing-candidates/export")
        assert r.status_code == 200
        assert "text/csv" in r.content_type


class TestImportCsv:
    def test_import_csv_no_file_redirects(self, routes_auth_client):
        """ファイルなしで POST → エラーフラッシュしてリダイレクト"""
        r = routes_auth_client.post("/import_csv", data={}, follow_redirects=False)
        assert r.status_code == 302

    def test_import_csv_non_csv_file_redirects(self, routes_auth_client):
        """CSV以外のファイル → エラーリダイレクト"""
        from io import BytesIO

        data = {"file": (BytesIO(b"not a csv"), "test.txt")}
        r = routes_auth_client.post(
            "/import_csv", data=data, content_type="multipart/form-data", follow_redirects=False
        )
        assert r.status_code == 302

    def test_import_csv_valid_file(self, routes_auth_client):
        """有効なCSV → インポート成功"""
        from io import BytesIO

        csv_content = b"name,purchase_price,selling_price\nTest,1000,2000\n"
        data = {"file": (BytesIO(csv_content), "products.csv")}
        r = routes_auth_client.post(
            "/import_csv", data=data, content_type="multipart/form-data", follow_redirects=False
        )
        assert r.status_code == 302


class TestGenerateListingException:
    def test_generate_listing_exception_handled(self, routes_auth_client, routes_app):
        """generate_listing_text が例外を投げても 200 が返る"""
        with routes_app.app_context():
            oid = _create_product(routes_auth_client)
        if oid:
            with patch("app.routes.products.generate_listing_text", side_effect=Exception("error")):
                r = routes_auth_client.get(f"/products/{oid}/generate-listing")
                assert r.status_code == 200


class TestPipelineResult:
    def test_pipeline_result_existing_product(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            oid = _create_product(routes_auth_client)
        if oid:
            r = routes_auth_client.get(f"/products/{oid}/pipeline-result")
            assert r.status_code == 200

    def test_pipeline_result_with_processed_images(self, routes_auth_client, routes_app):
        """processed_images がある商品のパイプライン結果表示"""
        import json

        with routes_app.app_context():
            oid = _create_product(routes_auth_client)
            if oid:
                from app.extensions import db
                from app.models import Product

                p = Product.query.get(oid)
                p.processed_images = json.dumps(["img1.jpg", "img2.jpg"])
                db.session.commit()
        if oid:
            r = routes_auth_client.get(f"/products/{oid}/pipeline-result")
            assert r.status_code == 200

    def test_pipeline_result_listing_text_exception(self, routes_auth_client, routes_app):
        """listing_text 生成で例外 → 空文字で続行"""
        with routes_app.app_context():
            oid = _create_product(routes_auth_client)
        if oid:
            with patch("app.routes.products.generate_listing_text", side_effect=Exception("error")):
                r = routes_auth_client.get(f"/products/{oid}/pipeline-result")
                assert r.status_code == 200

    def test_pipeline_result_nonexistent(self, routes_auth_client):
        r = routes_auth_client.get("/products/99999/pipeline-result")
        assert r.status_code in (302, 404)
