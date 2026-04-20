"""FR-010基盤: 顧客問い合わせAI ChatBotモデル"""

from __future__ import annotations

from datetime import datetime

from app.extensions import db


class CustomerInquiry(db.Model):
    """顧客問い合わせ: FAQマッチ→AI回答→人間承認フロー"""

    __tablename__ = "customer_inquiry"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    customer_name = db.Column(db.String(128), nullable=False, default="")
    customer_email = db.Column(db.String(256), nullable=False, default="")
    subject = db.Column(db.String(256), nullable=False)
    message = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(32), default="pending", nullable=False)
    response_text = db.Column(db.Text, nullable=True)
    faq_template_id = db.Column(
        db.Integer, db.ForeignKey("faq_template.id"), nullable=True
    )
    escalation_reason = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    def mark_matched(self, faq_template_id: int, response: str) -> None:
        self.status = "matched"
        self.faq_template_id = faq_template_id
        self.response_text = response

    def mark_approved(self) -> None:
        self.status = "approved"

    def mark_rejected(self, reason: str = "") -> None:
        self.status = "rejected"
        self.escalation_reason = reason

    def can_approve(self) -> bool:
        return self.status == "matched"

    def __repr__(self) -> str:
        return f"<CustomerInquiry id={self.id} status={self.status!r}>"
