"""Issue #63: auth.py カバレッジ 82→95%+

Tests for uncovered paths:
- L17-18: already authenticated user redirect
- L24-26: empty username/password validation
- L36-38: inactive user rejection
- L46: GET login page render
"""

from __future__ import annotations

import pytest
from werkzeug.security import generate_password_hash

from app import create_app
from app.extensions import db


@pytest.fixture()
def app_ctx():
    app = create_app()
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app_ctx):
    return app_ctx.test_client()


@pytest.fixture()
def create_user(app_ctx):
    def _make(username="testuser", password="testpass", is_active=True):
        from app.models.user import User

        user = User(
            username=username,
            password_hash=generate_password_hash(password),
            display_name="Test User",
            is_active=is_active,
        )
        db.session.add(user)
        db.session.commit()
        return user

    return _make


class TestAuthGetLogin:
    def test_get_login_page(self, client):
        resp = client.get("/auth/login")
        assert resp.status_code == 200


class TestAuthAlreadyAuthenticated:
    def test_authenticated_user_redirects_to_index(self, client, create_user):
        create_user()
        with client.session_transaction() as sess:
            sess["_user_id"] = "1"
        resp = client.get("/auth/login")
        assert resp.status_code == 302


class TestAuthEmptyValidation:
    def test_empty_username_returns_400(self, client):
        resp = client.post("/auth/login", data={"username": "", "password": "pass"})
        assert resp.status_code == 400

    def test_empty_password_returns_400(self, client):
        resp = client.post("/auth/login", data={"username": "user", "password": ""})
        assert resp.status_code == 400

    def test_whitespace_only_username_returns_400(self, client):
        resp = client.post("/auth/login", data={"username": "   ", "password": "pass"})
        assert resp.status_code == 400


class TestAuthInvalidCredentials:
    def test_wrong_password_returns_401(self, client, create_user):
        create_user()
        resp = client.post(
            "/auth/login", data={"username": "testuser", "password": "wrong"}
        )
        assert resp.status_code == 401

    def test_nonexistent_user_returns_401(self, client):
        resp = client.post(
            "/auth/login", data={"username": "nobody", "password": "pass"}
        )
        assert resp.status_code == 401


class TestAuthInactiveUser:
    def test_inactive_user_returns_403(self, client, create_user):
        create_user(is_active=False)
        resp = client.post(
            "/auth/login", data={"username": "testuser", "password": "testpass"}
        )
        assert resp.status_code == 403


class TestAuthSuccessfulLogin:
    def test_valid_login_redirects(self, client, create_user):
        create_user()
        resp = client.post(
            "/auth/login",
            data={"username": "testuser", "password": "testpass"},
            follow_redirects=False,
        )
        assert resp.status_code == 302

    def test_login_with_next_param(self, client, create_user):
        create_user()
        resp = client.post(
            "/auth/login?next=/manage",
            data={"username": "testuser", "password": "testpass"},
            follow_redirects=False,
        )
        assert resp.status_code == 302
        assert "/manage" in resp.headers["Location"]
