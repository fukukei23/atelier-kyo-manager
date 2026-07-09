"""Tests for app/services/chatbot_service.py"""

from unittest.mock import MagicMock, patch

import pytest

from app import create_app
from app.extensions import db


@pytest.fixture(scope="function")
def app():
    application = create_app()
    application.config.update(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
            "WTF_CSRF_ENABLED": False,
            "SECRET_KEY": "test-secret",
        }
    )
    with application.app_context():
        db.create_all()
        yield application
        db.session.remove()
        db.drop_all()


class TestChatBotServiceFaqMatch:
    def test_faq_match_found(self, app):
        with app.app_context():
            from app.models.faq_template import FaqTemplate
            from app.services.chatbot_service import ChatBotService

            ft = FaqTemplate(
                question_pattern="return,refund",
                answer_template="We accept returns within 30 days.",
                is_active=True,
            )
            db.session.add(ft)
            db.session.commit()

            result = ChatBotService._find_faq_match("return policy", "How do I return an item?")
            assert result is not None
            assert result.id == ft.id

    def test_faq_match_inactive_skipped(self, app):
        with app.app_context():
            from app.models.faq_template import FaqTemplate
            from app.services.chatbot_service import ChatBotService

            ft = FaqTemplate(
                question_pattern="return,refund",
                answer_template="We accept returns.",
                is_active=False,
            )
            db.session.add(ft)
            db.session.commit()

            result = ChatBotService._find_faq_match("return policy", "How do I return?")
            assert result is None

    def test_faq_match_no_templates(self, app):
        with app.app_context():
            from app.services.chatbot_service import ChatBotService

            result = ChatBotService._find_faq_match("anything", "some message")
            assert result is None


class TestChatBotServiceClassify:
    def test_classify_returns_faq_when_matched(self, app):
        with app.app_context():
            from app.models.faq_template import FaqTemplate
            from app.services.chatbot_service import ChatBotService

            ft = FaqTemplate(
                question_pattern="shipping,delivery",
                answer_template="Ships in 3-5 days.",
                is_active=True,
            )
            db.session.add(ft)
            db.session.commit()

            inquiry = MagicMock()
            inquiry.subject = "shipping question"
            inquiry.message = "When will my delivery arrive?"
            result = ChatBotService.classify_inquiry(inquiry)
            assert result == "faq"

    @patch("app.services.chatbot_service.AILlmController")
    def test_classify_returns_escalation(self, mock_llm, app):
        with app.app_context():
            from app.services.chatbot_service import ChatBotService

            mock_llm.quick.return_value = "escalation"
            inquiry = MagicMock()
            inquiry.subject = "Complaint about damaged goods"
            inquiry.message = "My item was broken!"
            result = ChatBotService.classify_inquiry(inquiry)
            assert result == "escalation"

    @patch("app.services.chatbot_service.AILlmController")
    def test_classify_returns_ai_on_llm_failure(self, mock_llm, app):
        with app.app_context():
            from app.services.chatbot_service import ChatBotService

            mock_llm.quick.side_effect = Exception("LLM error")
            inquiry = MagicMock()
            inquiry.subject = "question"
            inquiry.message = "some message"
            result = ChatBotService.classify_inquiry(inquiry)
            assert result == "ai"


class TestChatBotServiceGenerateResponse:
    @patch("app.services.chatbot_service.AILlmController")
    def test_generate_returns_text(self, mock_llm, app):
        with app.app_context():
            from app.services.chatbot_service import ChatBotService

            mock_llm.quick.return_value = "Thank you for your inquiry."
            inquiry = MagicMock()
            inquiry.subject = "question"
            inquiry.message = "some message"
            result = ChatBotService.generate_ai_response(inquiry)
            assert result == "Thank you for your inquiry."

    @patch("app.services.chatbot_service.AILlmController")
    def test_generate_returns_fallback_on_error(self, mock_llm, app):
        with app.app_context():
            from app.services.chatbot_service import ChatBotService

            mock_llm.quick.side_effect = Exception("LLM unavailable")
            inquiry = MagicMock()
            inquiry.subject = "q"
            inquiry.message = "m"
            result = ChatBotService.generate_ai_response(inquiry)
            assert "失敗" in result


class TestChatBotServiceProcessInquiry:
    def test_process_with_faq_match(self, app):
        with app.app_context():
            from app.models.faq_template import FaqTemplate
            from app.services.chatbot_service import ChatBotService

            ft = FaqTemplate(
                question_pattern="size,sizing",
                answer_template="Size guide available on product page.",
                is_active=True,
            )
            db.session.add(ft)
            db.session.commit()

            inquiry = ChatBotService.process_inquiry(
                subject="size question",
                message="What size should I order?",
                customer_name="Alice",
                customer_email="alice@example.com",
            )
            assert inquiry.id is not None
            assert inquiry.status == "matched"

    @patch("app.services.chatbot_service.AILlmController")
    def test_process_ai_path(self, mock_llm, app):
        with app.app_context():
            from app.services.chatbot_service import ChatBotService

            mock_llm.quick.return_value = "ai"
            inquiry = ChatBotService.process_inquiry(
                subject="general question",
                message="Some question without FAQ match",
            )
            assert inquiry.id is not None
