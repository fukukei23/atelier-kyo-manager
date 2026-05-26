"""FR-010基盤: 顧客対応AI ChatBotサービス"""

from __future__ import annotations

import logging

from app.config.constants import MAX_MESSAGE_LENGTH, MAX_SUBJECT_LENGTH
from app.extensions import db
from app.models.customer_inquiry import CustomerInquiry
from app.models.faq_template import FaqTemplate
from app.utils.ai_llm_controller import AILlmController

logger = logging.getLogger(__name__)


class ChatBotService:
    """FAQマッチ→AI回答→エスカレーション判定"""

    @staticmethod
    def _find_faq_match(subject: str, message: str):
        """アクティブなFAQテンプレートからマッチを検索"""
        combined = f"{subject} {message}"
        for template in FaqTemplate.query.filter_by(is_active=True):
            if template.match(combined):
                return template
        return None

    @staticmethod
    def classify_inquiry(inquiry: CustomerInquiry) -> str:
        """問い合わせを分類: 'faq' / 'ai' / 'escalation'"""
        if ChatBotService._find_faq_match(inquiry.subject, inquiry.message):
            return "faq"

        try:
            safe_subject = inquiry.subject[:MAX_SUBJECT_LENGTH]
            safe_message = inquiry.message[:MAX_MESSAGE_LENGTH]
            prompt = (
                "以下の顧客問い合わせを分類してください。"
                "単純な質問なら 'ai' 、専門的な判断やクレーム・緊急対応が必要なら 'escalation' とだけ返答してください。\n"
                f"件名: {safe_subject}\n"
                f"本文: {safe_message}"
            )
            classification = AILlmController.quick(prompt).strip().lower()
            if "escalation" in classification:
                return "escalation"
        except Exception:
            logger.warning("AI分類に失敗、デフォルト分類を使用")

        return "ai"

    @staticmethod
    def generate_ai_response(inquiry: CustomerInquiry) -> str:
        """AIで返答テキストを生成"""
        try:
            safe_subject = inquiry.subject[:MAX_SUBJECT_LENGTH]
            safe_message = inquiry.message[:MAX_MESSAGE_LENGTH]
            prompt = (
                "BUYMA出品者として、以下の顧客問い合わせに丁寧かつ専門的に回答してください: "
                f"{safe_subject} - {safe_message}"
            )
            return AILlmController.quick(prompt)
        except Exception as e:
            logger.error("AI返答生成失敗: %s", e)
            return "AIによる回答生成に失敗しました。しばらくしてからお試しください。"

    @staticmethod
    def process_inquiry(
        subject: str,
        message: str,
        customer_name: str = "",
        customer_email: str = "",
    ) -> CustomerInquiry:
        """問い合わせを作成し、FAQ/AI/エスカレーションに振り分け"""
        inquiry = CustomerInquiry(
            customer_name=customer_name,
            customer_email=customer_email,
            subject=subject,
            message=message,
            status="pending",
        )
        db.session.add(inquiry)
        db.session.flush()

        # 1. FAQマッチ
        faq = ChatBotService._find_faq_match(subject, message)
        if faq:
            try:
                response = faq.render()
            except Exception:
                response = faq.answer_template
            inquiry.mark_matched(faq.id, response)
            db.session.commit()
            return inquiry

        # 2. AI分類
        category = ChatBotService.classify_inquiry(inquiry)

        if category == "ai":
            response = ChatBotService.generate_ai_response(inquiry)
            inquiry.status = "matched"
            inquiry.response_text = response
        else:
            # escalation
            inquiry.status = "pending"
            inquiry.escalation_reason = "AI判定によりエスカレーションされました。手動対応が必要です。"

        db.session.commit()
        return inquiry
