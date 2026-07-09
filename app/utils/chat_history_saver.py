from __future__ import annotations

import contextlib
import logging
import os
from datetime import timedelta

from app.config.constants import CHAT_HISTORY_RETENTION_DAYS
from app.core.timezone import _utcnow
from app.models.result_models import GenerateResult

logger = logging.getLogger(__name__)


def save_chat_history(prompt: str, result: GenerateResult) -> None:
    """チャット履歴をデータベースに保存（再起動後も履歴が保持される）"""
    try:
        from app import db
        from app.models import ChatHistory

        session_id = os.getenv("CHAT_SESSION_ID", "default")

        total_tokens = None
        if result.tokens:
            if isinstance(result.tokens, dict):
                total_tokens = result.tokens.get("input", 0) + result.tokens.get("output", 0)
                if total_tokens == 0:
                    total_tokens = result.tokens.get("prompt_tokens", 0) + result.tokens.get("completion_tokens", 0)
            elif isinstance(result.tokens, (int, float)):
                total_tokens = int(result.tokens)

        user_msg = ChatHistory(session_id=session_id, role="user", content=prompt, created_at=_utcnow())
        db.session.add(user_msg)

        assistant_msg = ChatHistory(
            session_id=session_id,
            role="assistant",
            content=result.text,
            model_family=result.model_family,
            tokens=total_tokens,
            cost_usd=result.cost_usd,
            created_at=_utcnow(),
        )
        db.session.add(assistant_msg)

        db.session.commit()

        # 7日経過した履歴を削除
        cutoff = _utcnow() - timedelta(days=CHAT_HISTORY_RETENTION_DAYS)
        ChatHistory.query.filter(ChatHistory.created_at < cutoff).delete()
        db.session.commit()

        logger.info(f"Chat history saved for session {session_id}")
    except ImportError:
        logger.debug("ChatHistory model not available, skipping database save")
    except Exception as e:
        logger.warning(f"Failed to save chat history: {e}")
        with contextlib.suppress(BaseException):
            db.session.rollback()
