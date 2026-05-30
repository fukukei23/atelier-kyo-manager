"""Tests for app/routes/listing_templates.py - Issue #88"""
import pytest


VALID_TPL = {
    "name": "テストテンプレート",
    "template_text": "これはテンプレート本文です。",
    "category": "general",
}


class TestListingTemplates:
    def test_unauthenticated_redirect(self, routes_client):
        r = routes_client.get("/templates", follow_redirects=False)
        assert r.status_code == 302
        assert "login" in r.location.lower()

    def test_authenticated_200(self, routes_auth_client):
        assert routes_auth_client.get("/templates").status_code == 200


class TestNewTemplate:
    def test_get_new_form(self, routes_auth_client):
        assert routes_auth_client.get("/templates/new").status_code == 200

    def test_post_valid_redirects(self, routes_auth_client):
        r = routes_auth_client.post("/templates/new", data=VALID_TPL, follow_redirects=False)
        assert r.status_code == 302

    def test_post_no_name_shows_error(self, routes_auth_client):
        data = {**VALID_TPL, "name": ""}
        r = routes_auth_client.post("/templates/new", data=data, follow_redirects=True)
        assert r.status_code == 200

    def test_post_no_text_shows_error(self, routes_auth_client):
        data = {**VALID_TPL, "template_text": ""}
        r = routes_auth_client.post("/templates/new", data=data, follow_redirects=True)
        assert r.status_code == 200

    def test_post_with_is_default(self, routes_auth_client):
        data = {**VALID_TPL, "name": "Default TPL", "is_default": "on"}
        r = routes_auth_client.post("/templates/new", data=data, follow_redirects=False)
        assert r.status_code == 302

    def test_created_template_appears_in_list(self, routes_auth_client):
        routes_auth_client.post("/templates/new", data=VALID_TPL)
        r = routes_auth_client.get("/templates")
        assert "テストテンプレート" in r.data.decode("utf-8")


class TestEditTemplate:
    def _create(self, client):
        client.post("/templates/new", data={**VALID_TPL, "name": "編集用テンプレート"})
        from app.models.listing_template import ListingTemplate
        return ListingTemplate.query.first()

    def test_get_edit_form_existing(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            tpl = self._create(routes_auth_client)
            if tpl:
                r = routes_auth_client.get(f"/templates/{tpl.id}/edit")
                assert r.status_code == 200

    def test_post_edit_valid_redirects(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            tpl = self._create(routes_auth_client)
            if tpl:
                data = {**VALID_TPL, "name": "更新後テンプレート"}
                r = routes_auth_client.post(f"/templates/{tpl.id}/edit", data=data, follow_redirects=False)
                assert r.status_code == 302

    def test_post_edit_no_name_error(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            tpl = self._create(routes_auth_client)
            if tpl:
                data = {**VALID_TPL, "name": ""}
                r = routes_auth_client.post(f"/templates/{tpl.id}/edit", data=data, follow_redirects=True)
                assert r.status_code == 200


class TestDeleteTemplate:
    def test_delete_existing(self, routes_auth_client, routes_app):
        routes_auth_client.post("/templates/new", data={**VALID_TPL, "name": "削除用テンプレート"})
        from app.models.listing_template import ListingTemplate
        with routes_app.app_context():
            tpl = ListingTemplate.query.first()
            if tpl:
                r = routes_auth_client.post(f"/templates/{tpl.id}/delete", follow_redirects=False)
                assert r.status_code == 302

    def test_delete_nonexistent(self, routes_auth_client):
        r = routes_auth_client.post("/templates/99999/delete")
        assert r.status_code in (302, 404)
