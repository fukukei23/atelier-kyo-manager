from __future__ import annotations

from urllib.parse import urlparse

from app.core.timezone import _utcnow
from app.extensions import db


class ProhibitedSource(db.Model):
    __tablename__ = "prohibited_source"

    id = db.Column(db.Integer, primary_key=True)
    domain = db.Column(db.String(255), nullable=False, unique=True)
    reason = db.Column(db.Text, nullable=True)
    severity = db.Column(db.String(16), default="blocked")  # warning / blocked
    source_type = db.Column(db.String(16), default="domestic")  # domestic / overseas
    created_at = db.Column(db.DateTime, default=_utcnow)

    @staticmethod
    def is_prohibited(url: str, source_type: str = "domestic") -> tuple[bool, str]:
        """URL が禁止買付先かチェック。戻り値: (prohibited, reason)"""
        if not url:
            return False, ""
        try:
            host = urlparse(url).hostname or ""
        except Exception:
            return False, ""
        host = host.lower()
        entries = ProhibitedSource.query.filter_by(source_type=source_type).all()
        for entry in entries:
            ed = entry.domain.lower()
            if host == ed or host.endswith("." + ed):
                return True, entry.reason or f"{entry.domain} は禁止買付先です"
        return False, ""

    def __repr__(self) -> str:
        return f"<ProhibitedSource {self.domain} [{self.severity}]>"
