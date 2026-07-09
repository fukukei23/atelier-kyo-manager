"""FR-010基盤: FAQテンプレート返答モデル"""

from app.core.timezone import _utcnow
from app.extensions import db


class FaqTemplate(db.Model):
    """FAQテンプレート: キーワードマッチ→定型返答"""

    __tablename__ = "faq_template"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    category = db.Column(db.String(50), nullable=False, default="general")
    question_pattern = db.Column(db.String(200), nullable=False)
    answer_template = db.Column(db.Text, nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=_utcnow)
    updated_at = db.Column(db.DateTime, default=_utcnow, onupdate=_utcnow)

    def match(self, text: str) -> bool:
        """テキストにキーワードが含まれているか"""
        if not text or not self.question_pattern:
            return False
        keywords = [kw.strip().lower() for kw in self.question_pattern.split(",") if kw.strip()]
        lower_text = text.lower()
        return any(kw in lower_text for kw in keywords)

    def render(self, **kwargs) -> str:
        """テンプレート変数を置換して返す。未定義プレースホルダーは空文字にフォールバック。"""
        return self.answer_template.format_map({k: "" for k in _extract_placeholders(self.answer_template)} | kwargs)


def _extract_placeholders(template: str) -> set[str]:
    """テンプレート文字列から {name} プレースホルダーを抽出する。"""
    import re

    return set(re.findall(r"\{(\w+)\}", template))
