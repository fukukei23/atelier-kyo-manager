import pytest
from unittest.mock import patch, MagicMock
from flask import Flask

from app.decorators import handle_db, admin_required


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
