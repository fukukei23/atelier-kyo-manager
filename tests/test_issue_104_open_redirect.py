"""Issue #104: Open Redirect (CWE-601) 対策

app/auth.py login() の next パラメータ無検証 redirect を封じる:
- 相対パス（/ で始まり netloc/scheme を持たない）のみ許可
- 絶対URL・プロトコル相対（//evil.com）・バックスラッシュ（/\\evil.com）は
  トップページへフォールバック
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
    from app.models.user import User

    def _create(username="testuser", password="testpass"):
        user = User(
            username=username,
            password_hash=generate_password_hash(password),
            display_name="Test User",
            is_active=True,
        )
        db.session.add(user)
        db.session.commit()
        return user

    return _create


LOGIN = {"username": "testuser", "password": "testpass"}


def _login(client, next_value):
    return client.post(f"/auth/login?next={next_value}", data=LOGIN, follow_redirects=False)


class TestSafeNextRedirect:
    """安全なnext（相対パス）は許可する"""

    def test_relative_next_allowed(self, client, create_user):
        create_user()
        resp = _login(client, "/items")
        assert resp.status_code == 302
        assert resp.headers["Location"].endswith("/items")

    def test_next_with_query_allowed(self, client, create_user):
        create_user()
        resp = _login(client, "/items%3Fpage%3D2")
        assert resp.status_code == 302
        assert "page=2" in resp.headers["Location"]


class TestOpenRedirectBlocked:
    """危険なnextはトップページへフォールバック"""

    @pytest.mark.parametrize(
        "evil",
        [
            "https://evil.com",  # 絶対URL（https）
            "http://evil.com",  # 絶対URL（http）
            "//evil.com",  # プロトコル相対
            "%2F%2Fevil.com",  # URLエンコード済みプロトコル相対
            "/\\evil.com",  # バックスラッシュ（ブラウザ正規化で//化）
            "https:evil.com",  # スキーム相対変形
        ],
    )
    def test_evil_next_falls_back_to_index(self, client, create_user, evil):
        create_user()
        resp = _login(client, evil)
        assert resp.status_code == 302
        location = resp.headers["Location"]
        assert "evil.com" not in location
        # トップページ（main.index）へ
        assert location.endswith("/") or "index" in location

    def test_empty_next_falls_back_to_index(self, client, create_user):
        """next未指定は従来どおりトップへ"""
        create_user()
        resp = client.post("/auth/login", data=LOGIN, follow_redirects=False)
        assert resp.status_code == 302
