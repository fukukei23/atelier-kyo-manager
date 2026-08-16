"""Issue #105: セッションCookie保護とSECRET_KEYガード

- SESSION_COOKIE_HTTPONLY / SAMESITE(Lax) 常時設定
- SESSION_COOKIE_SECURE は staging/prod のみ True（ローカルhttp破壊回避）
- デフォルトSECRET_KEYは test 以外（staging/prod）で起動拒否
"""

from __future__ import annotations

import importlib

import pytest
from werkzeug.security import generate_password_hash

from app import create_app
from app.extensions import db

DEV_KEY = "dev-secret-key-change-in-production"


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


class TestCookieFlags:
    def test_cookie_httponly_and_samesite_on_login(self, client, create_user):
        """ログイン時のSet-CookieにHttpOnly・SameSite=Laxが付く"""
        create_user()
        resp = client.post(
            "/auth/login",
            data={"username": "testuser", "password": "testpass"},
            follow_redirects=False,
        )
        assert resp.status_code == 302
        cookie_headers = [v for k, v in resp.headers.items() if k == "Set-Cookie"]
        assert cookie_headers, "Set-Cookie が無い"
        joined = "; ".join(cookie_headers)
        assert "HttpOnly" in joined
        assert "SameSite=Lax" in joined

    def test_config_flags_defaults(self, app_ctx):
        """設定値: HttpOnly=True・Samesite=Lax・test段階ではSecure=False"""
        assert app_ctx.config["SESSION_COOKIE_HTTPONLY"] is True
        assert app_ctx.config["SESSION_COOKIE_SAMESITE"] == "Lax"
        assert app_ctx.config["SESSION_COOKIE_SECURE"] is False
        assert app_ctx.config["REMEMBER_COOKIE_HTTPONLY"] is True


class TestSecretKeyGuard:
    def _reload_config_with(self, monkeypatch, stage, secret=None):
        import app.config.config as config_mod

        monkeypatch.setenv("AK_STAGE", stage)
        if secret is None:
            monkeypatch.delenv("SECRET_KEY", raising=False)
        else:
            monkeypatch.setenv("SECRET_KEY", secret)
        try:
            return importlib.reload(config_mod)
        finally:
            # 呼び出し側で再度使えるよう状態は呼び出し後に戻す（呼び出し側責任）
            pass

    def test_prod_with_default_key_raises(self, monkeypatch):
        import app.config.config as config_mod

        monkeypatch.setenv("AK_STAGE", "prod")
        monkeypatch.delenv("SECRET_KEY", raising=False)
        with pytest.raises(RuntimeError, match="SECRET_KEY"):
            importlib.reload(config_mod)
        monkeypatch.setenv("AK_STAGE", "test")

    def test_staging_with_default_key_raises(self, monkeypatch):
        """staging もデフォルトキーは拒否（ISSUE-105のガード穴修正）"""
        import app.config.config as config_mod

        monkeypatch.setenv("AK_STAGE", "staging")
        monkeypatch.delenv("SECRET_KEY", raising=False)
        with pytest.raises(RuntimeError, match="SECRET_KEY"):
            importlib.reload(config_mod)
        monkeypatch.setenv("AK_STAGE", "test")

    def test_prod_with_explicit_key_ok(self, monkeypatch):
        import app.config.config as config_mod

        monkeypatch.setenv("AK_STAGE", "prod")
        monkeypatch.setenv("SECRET_KEY", "proper-production-key-xyz")
        # prod判定のCeleryガード（rediss/パスワード付きredis必須）も満たす
        monkeypatch.setenv("CELERY_BROKER_URL", "rediss://example:6379/0")
        monkeypatch.setenv("CELERY_RESULT_BACKEND", "rediss://example:6379/0")
        try:
            importlib.reload(config_mod)
            assert config_mod.AppConfig.SECRET_KEY == "proper-production-key-xyz"
            assert config_mod.AppConfig.SESSION_COOKIE_SECURE is True
        finally:
            monkeypatch.setenv("AK_STAGE", "test")
            monkeypatch.delenv("SECRET_KEY", raising=False)
            monkeypatch.delenv("CELERY_BROKER_URL", raising=False)
            monkeypatch.delenv("CELERY_RESULT_BACKEND", raising=False)
            importlib.reload(config_mod)
