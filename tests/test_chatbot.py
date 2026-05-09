"""Tests for CustomerInquiry model, ChatBotService, and routes (Issue #19 / FR-010)."""

import sys
from unittest.mock import patch, MagicMock

import pytest

from app import create_app
from app.extensions import db as _db
from app.models.customer_inquiry import CustomerInquiry
from app.models.faq_template import FaqTemplate

# AILlmController のモック: opentelemetry等が未インストールでもテスト可能にする
_mock_llm = MagicMock()
_mock_llm.quick = MagicMock()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app():
    app = create_app()
    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "test-secret"
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["LOGIN_DISABLED"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    with app.app_context():
        _db.create_all()
        yield app
        _db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def clean_db(app):
    """Clear tables between tests."""
    with app.app_context():
        CustomerInquiry.query.delete()
        FaqTemplate.query.delete()
        _db.session.commit()
    yield
    with app.app_context():
        CustomerInquiry.query.delete()
        FaqTemplate.query.delete()
        _db.session.commit()


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestCustomerInquiryModel:
    def test_mark_matched(self, app):
        with app.app_context():
            inquiry = CustomerInquiry(subject="test", message="msg")
            inquiry.mark_matched(1, "FAQ回答")
            assert inquiry.status == "matched"
            assert inquiry.faq_template_id == 1
            assert inquiry.response_text == "FAQ回答"

    def test_mark_approved(self, app):
        with app.app_context():
            inquiry = CustomerInquiry(subject="test", message="msg", status="matched")
            inquiry.mark_approved()
            assert inquiry.status == "approved"

    def test_mark_rejected(self, app):
        with app.app_context():
            inquiry = CustomerInquiry(subject="test", message="msg", status="matched")
            inquiry.mark_rejected(reason="不適切")
            assert inquiry.status == "rejected"
            assert inquiry.escalation_reason == "不適切"

    def test_mark_rejected_no_reason(self, app):
        with app.app_context():
            inquiry = CustomerInquiry(subject="test", message="msg", status="matched")
            inquiry.mark_rejected()
            assert inquiry.status == "rejected"
            assert inquiry.escalation_reason == ""

    def test_can_approve(self, app):
        with app.app_context():
            assert CustomerInquiry(status="matched").can_approve() is True
            assert CustomerInquiry(status="pending").can_approve() is False
            assert CustomerInquiry(status="approved").can_approve() is False
            assert CustomerInquiry(status="rejected").can_approve() is False


# ---------------------------------------------------------------------------
# Service tests
# ---------------------------------------------------------------------------

class TestChatBotService:
    def test_process_inquiry_faq_match(self, app, clean_db):
        """FAQマッチ時にmatchedステータスになる"""
        with app.app_context():
            faq = FaqTemplate(
                category="shipping",
                question_pattern="配送,届く",
                answer_template="通常3〜5日でお届けします。",
                is_active=True,
            )
            _db.session.add(faq)
            _db.session.commit()

            from app.services.chatbot_service import ChatBotService
            inquiry = ChatBotService.process_inquiry(
                subject="配送について",
                message="いつ届きますか？",
            )
            assert inquiry.status == "matched"
            assert inquiry.faq_template_id == faq.id
            assert "3〜5日" in inquiry.response_text

    def test_process_inquiry_ai_fallback(self, app, clean_db):
        """FAQ不一致時にAI回答が生成される"""
        _mock_llm.quick.return_value = "AIによるテスト回答です。"
        with app.app_context():
            with patch("app.services.chatbot_service.AILlmController", _mock_llm):
                from app.services.chatbot_service import ChatBotService
                inquiry = ChatBotService.process_inquiry(
                    subject="カスタム質問",
                    message="これはFAQにない質問です",
                )
                assert inquiry.status == "matched"
                assert "テスト回答" in inquiry.response_text

    def test_process_inquiry_escalation(self, app, clean_db):
        """escalation判定時にpendingのまま"""
        _mock_llm.quick.return_value = "escalation"
        with app.app_context():
            with patch("app.services.chatbot_service.AILlmController", _mock_llm):
                from app.services.chatbot_service import ChatBotService
                inquiry = ChatBotService.process_inquiry(
                    subject="クレーム",
                    message="商品が破損していました。すぐに対応してください。",
                )
                assert inquiry.status == "pending"
                assert inquiry.escalation_reason is not None

    def test_classify_inquiry_faq(self, app, clean_db):
        """FAQマッチ時にfaqを返す"""
        with app.app_context():
            faq = FaqTemplate(
                question_pattern="返品",
                answer_template="返品は7日以内に対応します。",
                is_active=True,
            )
            _db.session.add(faq)
            _db.session.commit()

            from app.services.chatbot_service import ChatBotService
            inquiry = CustomerInquiry(subject="返品したい", message="返品方法を教えて")
            assert ChatBotService.classify_inquiry(inquiry) == "faq"

    def test_generate_ai_response(self, app, clean_db):
        """AI返答生成が呼ばれる"""
        _mock_llm.quick.return_value = "AI回答テキスト"
        with app.app_context():
            with patch("app.services.chatbot_service.AILlmController", _mock_llm):
                from app.services.chatbot_service import ChatBotService
                inquiry = CustomerInquiry(subject="質問", message="本文")
                result = ChatBotService.generate_ai_response(inquiry)
                assert result == "AI回答テキスト"

    def test_generate_ai_response_error(self, app, clean_db):
        """AI失敗時にエラーメッセージを返す"""
        _mock_llm.quick.side_effect = RuntimeError("API Error")
        with app.app_context():
            with patch("app.services.chatbot_service.AILlmController", _mock_llm):
                from app.services.chatbot_service import ChatBotService
                inquiry = CustomerInquiry(subject="質問", message="本文")
                result = ChatBotService.generate_ai_response(inquiry)
                assert "失敗" in result


# ---------------------------------------------------------------------------
# Route tests
# ---------------------------------------------------------------------------

class TestChatBotRoutes:
    def test_dashboard(self, client, clean_db):
        resp = client.get("/chatbot")
        assert resp.status_code == 200
        assert "ChatBot" in resp.text

    def test_dashboard_with_filter(self, client, clean_db):
        resp = client.get("/chatbot?status=pending")
        assert resp.status_code == 200

    def test_create_form(self, client, clean_db):
        resp = client.get("/chatbot/create")
        assert resp.status_code == 200
        assert "問い合わせ" in resp.text

    def test_create_submit_faq(self, app, client, clean_db):
        """FAQマッチする問い合わせをPOST"""
        with app.app_context():
            faq = FaqTemplate(
                question_pattern="配送",
                answer_template="配送は3日です。",
                is_active=True,
            )
            _db.session.add(faq)
            _db.session.commit()

        resp = client.post("/chatbot/create", data={
            "subject": "配送について",
            "message": "配送日を教えてください",
            "customer_name": "テスト太郎",
            "customer_email": "test@example.com",
        }, follow_redirects=True)
        assert resp.status_code == 200
        with app.app_context():
            inquiry = CustomerInquiry.query.first()
            assert inquiry is not None
            assert inquiry.subject == "配送について"
            assert inquiry.customer_name == "テスト太郎"

    def test_create_submit_validation(self, client, clean_db):
        """件名・本文なしはエラー"""
        resp = client.post("/chatbot/create", data={
            "subject": "",
            "message": "",
        })
        assert resp.status_code == 200
        assert "必須" in resp.text

    def test_detail_page(self, app, client, clean_db):
        with app.app_context():
            inquiry = CustomerInquiry(
                subject="テスト件名", message="テスト本文", status="matched",
                response_text="AI回答プレビュー",
            )
            _db.session.add(inquiry)
            _db.session.commit()
            iid = inquiry.id

        resp = client.get(f"/chatbot/{iid}")
        assert resp.status_code == 200
        assert "テスト件名" in resp.text
        assert "AI回答プレビュー" in resp.text

    def test_detail_404(self, client, clean_db):
        resp = client.get("/chatbot/99999")
        assert resp.status_code == 404

    def test_approve(self, app, client, clean_db):
        with app.app_context():
            inquiry = CustomerInquiry(
                subject="テスト", message="msg", status="matched",
                response_text="回答",
            )
            _db.session.add(inquiry)
            _db.session.commit()
            iid = inquiry.id

        resp = client.post(f"/chatbot/{iid}/approve", follow_redirects=True)
        assert resp.status_code == 200
        with app.app_context():
            inquiry = CustomerInquiry.query.get(iid)
            assert inquiry.status == "approved"

    def test_approve_not_matched(self, app, client, clean_db):
        """matched以外のステータスは承認できない"""
        with app.app_context():
            inquiry = CustomerInquiry(subject="テスト", message="msg", status="pending")
            _db.session.add(inquiry)
            _db.session.commit()
            iid = inquiry.id

        resp = client.post(f"/chatbot/{iid}/approve", follow_redirects=True)
        assert resp.status_code == 200
        with app.app_context():
            inquiry = CustomerInquiry.query.get(iid)
            assert inquiry.status == "pending"

    def test_reject(self, app, client, clean_db):
        with app.app_context():
            inquiry = CustomerInquiry(
                subject="テスト", message="msg", status="matched",
                response_text="回答",
            )
            _db.session.add(inquiry)
            _db.session.commit()
            iid = inquiry.id

        resp = client.post(f"/chatbot/{iid}/reject",
                           data={"reason": "不適切"},
                           follow_redirects=True)
        assert resp.status_code == 200
        with app.app_context():
            inquiry = CustomerInquiry.query.get(iid)
            assert inquiry.status == "rejected"
            assert inquiry.escalation_reason == "不適切"
