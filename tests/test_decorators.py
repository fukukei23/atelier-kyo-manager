from unittest.mock import patch

import pytest
from flask import Flask

from app.core.pricing.schemas import PriceIntegrityError
from app.decorators import admin_required, handle_db
from app.utils.decorators import handle_db_error


def create_test_app():
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "test"

    @app.route("/test_success")
    @handle_db(success_msg="成功", redirect_endpoint="index")
    def test_success_route():
        return "OK"

    @app.route("/test_error_default")
    @handle_db(error_msg="カスタムエラー", redirect_endpoint="index")
    def test_error_default():
        raise ValueError("DBの障害")

    @app.route("/test_error_no_endpoint")
    @handle_db()
    def test_error_no_endpoint():
        raise Exception("想定外のエラー")

    @app.route("/test_price_integrity_propagates")
    @handle_db_error()
    def test_price_integrity_route():
        raise PriceIntegrityError("信頼性エラー")

    @app.route("/test_db_error_caught")
    @handle_db_error("index")
    def test_db_error_route():
        raise RuntimeError("DB操作失敗")

    @app.route("/test_admin")
    @admin_required
    def test_admin_route():
        return "Admin OK"

    @app.route("/index")
    def index():
        return "Index"

    return app


@pytest.fixture
def app():
    return create_test_app()


@pytest.fixture
def client(app):
    return app.test_client()


class TestHandleDbDecorator:
    @patch("app.decorators.db")
    def test_success_path(self, mock_db, app, client):
        with app.test_request_context():
            response = client.get("/test_success")
            assert response.status_code == 200
            assert response.data == b"OK"
            mock_db.session.rollback.assert_not_called()

    @patch("app.decorators.db")
    def test_exception_rollback_redirect_no_endpoint(self, mock_db, app, client):
        with app.test_request_context():
            response = client.get("/test_error_no_endpoint")
            assert response.status_code == 302
            mock_db.session.rollback.assert_called_once()
            with client.session_transaction() as session:
                flashes = session.get("_flashes", [])
                assert len(flashes) == 1
                category, message = flashes[0]
                assert category == "error"

    @patch("app.decorators.db")
    def test_exception_with_custom_error_msg(self, mock_db, app, client):
        with app.test_request_context():
            response = client.get("/test_error_default")
            assert response.status_code == 302
            assert "/index" in response.headers["Location"]
            mock_db.session.rollback.assert_called_once()
            with client.session_transaction() as session:
                flashes = session.get("_flashes", [])
                assert len(flashes) == 1
                category, message = flashes[0]
                assert message == "カスタムエラー"


class TestHandleDbErrorDecorator:
    """ISSUE-102: handle_db_error は PriceIntegrityError を握りつぶさず伝播。"""

    @patch("app.utils.decorators.db")
    def test_price_integrity_error_propagates(self, mock_db, app, client):
        """PriceIntegrityError は伝播し redirect(302) で握りつぶされない。"""
        with app.test_request_context():
            # TESTING=True なので伝播した例外は raise される
            with pytest.raises(PriceIntegrityError):
                client.get("/test_price_integrity_propagates")
            mock_db.session.rollback.assert_called_once()

    @patch("app.utils.decorators.db")
    def test_db_error_caught_redirected(self, mock_db, app, client):
        """通常のDB例外(Exception) は従来通り flash+redirect(302) で処理（回帰維持）。"""
        with app.test_request_context():
            response = client.get("/test_db_error_caught")
            assert response.status_code == 302
            mock_db.session.rollback.assert_called_once()


class TestAdminRequiredDecorator:
    @patch("app.decorators.current_user")
    def test_not_admin_abort_403(self, mock_current_user, app, client):
        mock_current_user.is_authenticated = True
        mock_current_user.is_admin = False
        with app.test_request_context():
            response = client.get("/test_admin")
            assert response.status_code == 403

    @patch("app.decorators.current_user")
    def test_passes_when_admin(self, mock_current_user, app, client):
        mock_current_user.is_authenticated = True
        mock_current_user.is_admin = True
        with app.test_request_context():
            response = client.get("/test_admin")
            assert response.status_code == 200
            assert response.data == b"Admin OK"
