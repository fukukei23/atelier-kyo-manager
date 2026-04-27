import pytest
from unittest.mock import patch, MagicMock
from app import create_app


@pytest.fixture
def app():
    app = create_app()
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    return app


@pytest.fixture
def client(app):
    return app.test_client()


# L47: GET /auth/login → ログインページ表示
def test_get_login_returns_200(client):
    response = client.get("/auth/login")
    assert response.status_code == 200


# L20: 認証済ユーザー → メインページへリダイレクト
def test_post_login_authenticated_redirect(client):
    with patch("app.auth.current_user") as mock_cu:
        mock_cu.is_authenticated = True
        resp = client.post("/auth/login", data={"username": "u", "password": "p"})
        assert resp.status_code == 302


# L27-28: 空ユーザー名/パスワード → 400
def test_post_login_empty_fields(client):
    with patch("app.auth.current_user") as mock_cu:
        mock_cu.is_authenticated = False
        resp = client.post("/auth/login", data={"username": "", "password": "p"})
        assert resp.status_code == 400

        resp = client.post("/auth/login", data={"username": "u", "password": ""})
        assert resp.status_code == 400


# ユーザーなし → 401
def test_post_login_user_not_found(client):
    with patch("app.auth.current_user") as mock_cu, \
         patch("app.models.user.User") as MockUser:
        mock_cu.is_authenticated = False
        MockUser.query.filter_by.return_value.first.return_value = None
        resp = client.post("/auth/login", data={"username": "x", "password": "p"})
        assert resp.status_code == 401


# パスワード間違い → 401
def test_post_login_wrong_password(client):
    with patch("app.auth.current_user") as mock_cu, \
         patch("app.models.user.User") as MockUser:
        mock_cu.is_authenticated = False
        mock_user = MagicMock()
        mock_user.check_password.return_value = False
        MockUser.query.filter_by.return_value.first.return_value = mock_user
        resp = client.post("/auth/login", data={"username": "u", "password": "wrong"})
        assert resp.status_code == 401


# L38-39: 非アクティブユーザー → 403
def test_post_login_inactive_user(client):
    with patch("app.auth.current_user") as mock_cu, \
         patch("app.models.user.User") as MockUser:
        mock_cu.is_authenticated = False
        mock_user = MagicMock()
        mock_user.check_password.return_value = True
        mock_user.is_active = False
        MockUser.query.filter_by.return_value.first.return_value = mock_user
        resp = client.post("/auth/login", data={"username": "u", "password": "p"})
        assert resp.status_code == 403


# 正常ログイン → リダイレクト
def test_post_login_success(client):
    with patch("app.auth.current_user") as mock_cu, \
         patch("app.models.user.User") as MockUser, \
         patch("app.auth.login_user") as mock_login:
        mock_cu.is_authenticated = False
        mock_user = MagicMock()
        mock_user.check_password.return_value = True
        mock_user.is_active = True
        mock_user.display_name = "テスト"
        mock_user.username = "testuser"
        MockUser.query.filter_by.return_value.first.return_value = mock_user
        resp = client.post("/auth/login", data={"username": "testuser", "password": "p"})
        assert resp.status_code == 302
        mock_login.assert_called_once_with(mock_user, remember=True)
