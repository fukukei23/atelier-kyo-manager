"""atelier-kyo-manager ルートテスト"""

from unittest.mock import MagicMock, patch

import pytest

from app import create_app
from app.extensions import db
from app.models.user import User


@pytest.fixture(scope="function")
def app():
    app = create_app()
    app.config.update(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
            "WTF_CSRF_ENABLED": False,
            "SECRET_KEY": "test-secret",
        }
    )
    with app.app_context():
        db.create_all()
        yield app
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
def auth_client(client, app, test_user):
    client.post("/auth/login", data=test_user, follow_redirects=False)
    return client


# ---- Auth ----
class TestAuth:
    def test_login_success(self, client, test_user):
        r = client.post("/auth/login", data=test_user, follow_redirects=False)
        assert r.status_code == 302

    def test_login_wrong_password(self, client, test_user):
        r = client.post("/auth/login", data={"username": "testuser", "password": "wrong"})
        assert r.status_code == 401

    def test_login_nonexistent_user(self, client):
        r = client.post("/auth/login", data={"username": "nobody", "password": "x"})
        assert r.status_code == 401

    def test_logout(self, auth_client):
        r = auth_client.get("/auth/logout", follow_redirects=False)
        assert r.status_code == 302


# ---- Protected routes redirect ----
class TestProtectedRoutes:
    @pytest.mark.parametrize(
        "path",
        [
            "/manage",
            "/products",
            "/orders",
            "/partners",
            "/stock-check",
            "/popularity",
            "/regions",
            "/customers",
            "/cashflow",
            "/brand-analytics",
            "/listing-progress",
            "/templates",
            "/auto-research",
        ],
    )
    def test_unauthenticated_redirects(self, client, path):
        r = client.get(path, follow_redirects=False)
        assert r.status_code == 302
        assert "/auth/login" in r.headers.get("Location", "")


# ---- Authenticated pages 200 ----
class TestAuthenticatedPages:
    @pytest.mark.parametrize(
        "path",
        [
            "/",
            "/manage",
            "/products",
            "/orders",
            "/partners",
            "/stock-check",
            "/popularity",
            "/regions",
            "/customers",
            "/cashflow",
            "/brand-analytics",
            "/listing-progress",
            "/templates",
            "/auto-research",
        ],
    )
    def test_page_200(self, auth_client, path):
        r = auth_client.get(path)
        assert r.status_code == 200


# ---- Compat redirects ----
class TestCompatRedirects:
    def test_dashboard_redirect(self, auth_client):
        r = auth_client.get("/dashboard", follow_redirects=False)
        assert r.status_code == 200

    def test_listing_templates_redirect(self, auth_client):
        r = auth_client.get("/listing-templates", follow_redirects=False)
        assert r.status_code == 302
        assert "/templates" in r.headers["Location"]

    def test_region_recommendations_redirect(self, auth_client):
        r = auth_client.get("/region-recommendations", follow_redirects=False)
        assert r.status_code == 302
        assert "/regions" in r.headers["Location"]

    def test_repeat_customers_redirect(self, auth_client):
        r = auth_client.get("/repeat-customers", follow_redirects=False)
        assert r.status_code == 302
        assert "/customers" in r.headers["Location"]

    def test_auto_research_underscore(self, auth_client):
        r = auth_client.get("/auto_research", follow_redirects=False)
        assert r.status_code == 302
        assert "/auto-research" in r.headers["Location"]


# ---- Stock Check API ----
class TestStockCheckAPI:
    def test_fetch_all_empty(self, auth_client):
        r = auth_client.post("/api/stock-check/fetch-all")
        assert r.status_code == 200
        data = r.get_json()
        assert data["success"] is True

    def test_fetch_all_requires_auth(self, client):
        r = client.post("/api/stock-check/fetch-all", follow_redirects=False)
        assert r.status_code == 302


# ---- PriceScraper mock ----
class TestPriceScraper:
    def test_fetch_success(self):
        with patch("app.services.price_scraper.requests.Session.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = "<html><title>Test</title><span>¥12,345</span></html>"
            mock_resp.url = "http://example.com"
            mock_resp.headers = {}
            mock_get.return_value = mock_resp

            from app.services.price_scraper import PriceScraper

            scraper = PriceScraper()
            result = scraper.fetch("http://example.com")
            assert result["success"] is True
            assert result["price"] == 12345
            scraper.close()

    def test_fetch_connection_error(self):
        import requests as _req

        with patch("app.services.price_scraper.requests.Session.get") as mock_get:
            mock_get.side_effect = _req.ConnectionError("fail")

            from app.services.price_scraper import PriceScraper

            result = PriceScraper().fetch("http://bad.example.com")
            assert result["success"] is False

    def test_fetch_sold_out(self):
        with patch("app.services.price_scraper.requests.Session.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = "<html><title>Sold</title><div>SOLD OUT</div></html>"
            mock_resp.url = "http://example.com"
            mock_resp.headers = {}
            mock_get.return_value = mock_resp

            from app.services.price_scraper import PriceScraper

            result = PriceScraper().fetch("http://example.com")
            assert result["in_stock"] is False


# ---- PriceScraper Fallback (FR-004) ----
class TestPriceScraperFallback:
    def test_classify_error_blocked(self):
        from app.services.price_scraper import PriceScraper

        assert PriceScraper.classify_error("403") == "blocked"
        assert PriceScraper.classify_error("429") == "blocked"
        assert PriceScraper.classify_error("Cloudflare") == "blocked"

    def test_classify_error_not_found(self):
        from app.services.price_scraper import PriceScraper

        assert PriceScraper.classify_error("404") == "not_found"

    def test_classify_error_timeout(self):
        from app.services.price_scraper import PriceScraper

        assert PriceScraper.classify_error("タイムアウト") == "timeout"
        assert PriceScraper.classify_error("Timeout") == "timeout"

    def test_classify_error_connection(self):
        from app.services.price_scraper import PriceScraper

        assert PriceScraper.classify_error("接続失敗") == "connection"

    def test_classify_error_server(self):
        from app.services.price_scraper import PriceScraper

        assert PriceScraper.classify_error("HTTP 500") == "server_error"

    def test_classify_error_unknown(self):
        from app.services.price_scraper import PriceScraper

        assert PriceScraper.classify_error("何か不明なエラー") == "unknown"

    def test_fetch_with_retry_success_first(self):
        from app.services.price_scraper import PriceScraper

        with patch("app.services.price_scraper.requests.Session.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = "<html><title>OK</title><span>¥9,999</span></html>"
            mock_resp.url = "http://example.com"
            mock_resp.headers = {}
            mock_get.return_value = mock_resp

            scraper = PriceScraper()
            result = scraper.fetch_with_retry("http://example.com")
            assert result["success"] is True
            assert result["retry_count"] == 0
            assert result.get("error_category") is None
            scraper.close()

    @patch("time.sleep", return_value=None)
    def test_fetch_with_retry_blocked_retries(self, mock_sleep):
        import requests as _req

        from app.services.price_scraper import PriceScraper

        with patch("app.services.price_scraper.requests.Session.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 403
            mock_resp.raise_for_status.side_effect = _req.HTTPError(response=mock_resp)
            mock_resp.text = "Forbidden"
            mock_resp.url = "http://example.com"
            mock_resp.headers = {}
            mock_get.return_value = mock_resp

            scraper = PriceScraper()
            result = scraper.fetch_with_retry("http://example.com", max_retries=3)
            assert result["success"] is False
            assert result["retry_count"] == 2
            assert result["error_category"] == "blocked"
            assert mock_get.call_count == 3
            assert mock_sleep.call_count == 2
            scraper.close()

    def test_fetch_with_retry_not_found_no_retry(self):
        import requests as _req

        from app.services.price_scraper import PriceScraper

        with patch("app.services.price_scraper.requests.Session.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 404
            mock_resp.raise_for_status.side_effect = _req.HTTPError(response=mock_resp)
            mock_resp.text = "Not Found"
            mock_resp.url = "http://example.com"
            mock_resp.headers = {}
            mock_get.return_value = mock_resp

            scraper = PriceScraper()
            result = scraper.fetch_with_retry("http://example.com")
            assert result["success"] is False
            assert result["retry_count"] == 0
            assert result["error_category"] == "not_found"
            assert mock_get.call_count == 1
            scraper.close()

    def test_fetch_cached_returns_cache(self):
        from app.services.price_scraper import PriceScraper

        with patch("app.services.price_scraper.requests.Session.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = "<html><title>C</title><span>¥1,000</span></html>"
            mock_resp.url = "http://example.com/cached"
            mock_resp.headers = {}
            mock_get.return_value = mock_resp

            scraper = PriceScraper()
            r1 = scraper.fetch_cached("http://example.com/cached")
            assert r1["success"] is True
            r2 = scraper.fetch_cached("http://example.com/cached")
            assert r2["success"] is True
            assert mock_get.call_count == 1  # キャッシュヒットで2回目は呼ばれない
            scraper.close()


# ---- NotificationService (FR-009基盤) ----
class TestNotificationService:
    def test_send_no_webhook_url(self):
        from app.services.notification_service import NotificationService

        ns = NotificationService()
        result = ns.send("test")
        assert result["success"] is False
        assert "未設定" in result["error"]

    def test_send_success(self):
        from app.services.notification_service import NotificationService

        with patch("app.services.notification_service.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            ns = NotificationService()
            ns.webhook_url = "https://hooks.slack.com/test"
            result = ns.send("Hello Slack")
            assert result["success"] is True
            mock_post.assert_called_once()

    def test_send_connection_error(self):
        import requests as _req

        from app.services.notification_service import NotificationService

        with patch("app.services.notification_service.requests.post") as mock_post:
            mock_post.side_effect = _req.ConnectionError("fail")

            ns = NotificationService()
            ns.webhook_url = "https://hooks.slack.com/test"
            result = ns.send("Hello")
            assert result["success"] is False

    def test_send_pipeline_result_success(self):
        from app.services.notification_service import NotificationService

        with patch("app.services.notification_service.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            ns = NotificationService()
            ns.webhook_url = "https://hooks.slack.com/test"
            result = ns.send_pipeline_result("TestProduct", "success", 12.5)
            assert result["success"] is True
            assert "✅" in mock_post.call_args[1]["json"]["text"]
            assert "TestProduct" in mock_post.call_args[1]["json"]["text"]

    def test_send_pipeline_result_failed(self):
        from app.services.notification_service import NotificationService

        with patch("app.services.notification_service.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            ns = NotificationService()
            ns.webhook_url = "https://hooks.slack.com/test"
            result = ns.send_pipeline_result("TestProduct", "failed", 5.0, errors="timeout")
            assert result["success"] is True
            assert "❌" in mock_post.call_args[1]["json"]["text"]

    def test_init_app_reads_config(self, app):
        from app.services.notification_service import NotificationService

        ns = NotificationService()
        ns.init_app(app)
        assert ns.webhook_url == ""  # test config has no SLACK_WEBHOOK_URL


# ---- FaqTemplate (FR-010基盤) ----
class TestFaqTemplateModel:
    def test_match_single_keyword(self):
        from app.models.faq_template import FaqTemplate

        faq = FaqTemplate(question_pattern="送料", answer_template="送料無料です")
        assert faq.match("送料はいくらですか") is True
        assert faq.match("返品したい") is False

    def test_match_multiple_keywords(self):
        from app.models.faq_template import FaqTemplate

        faq = FaqTemplate(question_pattern="送料, 配送, 届かない", answer_template="確認します")
        assert faq.match("配送方法は？") is True
        assert faq.match("商品が届かない") is True
        assert faq.match("サイズ") is False

    def test_match_empty_text(self):
        from app.models.faq_template import FaqTemplate

        faq = FaqTemplate(question_pattern="送料", answer_template="送料無料")
        assert faq.match("") is False
        assert faq.match(None) is False

    def test_render_template(self):
        from app.models.faq_template import FaqTemplate

        faq = FaqTemplate(question_pattern="送料", answer_template="{product_name}は送料無料です")
        result = faq.render(product_name="テスト商品")
        assert "テスト商品" in result
        assert "送料無料" in result


class TestFaqTemplateRoutes:
    def test_faq_templates_page_200(self, auth_client):
        r = auth_client.get("/faq-templates")
        assert r.status_code == 200

    def test_create_faq_template_form_200(self, auth_client):
        r = auth_client.get("/faq-templates/new")
        assert r.status_code == 200

    def test_create_faq_template_submit(self, auth_client, app):
        r = auth_client.post(
            "/faq-templates/new",
            data={
                "category": "shipping",
                "question_pattern": "送料,配送",
                "answer_template": "{product_name}は送料無料です",
            },
            follow_redirects=True,
        )
        assert r.status_code == 200
        with app.app_context():
            from app.models.faq_template import FaqTemplate

            faq = FaqTemplate.query.first()
            assert faq is not None
            assert faq.category == "shipping"
            assert faq.match("送料は？")

    def test_create_faq_template_missing_fields(self, auth_client):
        r = auth_client.post(
            "/faq-templates/new",
            data={
                "category": "general",
                "question_pattern": "",
                "answer_template": "",
            },
        )
        assert r.status_code == 200  # re-renders form

    def test_delete_faq_template(self, auth_client, app):
        with app.app_context():
            from app.models.faq_template import FaqTemplate

            faq = FaqTemplate(category="test", question_pattern="test", answer_template="test")
            db.session.add(faq)
            db.session.commit()
            fid = faq.id
        r = auth_client.post(f"/faq-templates/{fid}/delete", follow_redirects=True)
        assert r.status_code == 200
        with app.app_context():
            assert FaqTemplate.query.get(fid) is None

    def test_faq_match_api(self, auth_client, app):
        with app.app_context():
            from app.models.faq_template import FaqTemplate

            faq = FaqTemplate(
                category="shipping", question_pattern="送料,配送", answer_template="送料無料です", is_active=True
            )
            db.session.add(faq)
            db.session.commit()
        r = auth_client.post("/api/faq-match", json={"text": "送料はいくらですか"}, content_type="application/json")
        assert r.status_code == 200
        data = r.get_json()
        assert data["success"] is True
        assert len(data["matches"]) >= 1

    def test_faq_match_api_no_text(self, auth_client):
        r = auth_client.post("/api/faq-match", json={}, content_type="application/json")
        assert r.status_code == 400
