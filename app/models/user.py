# ======================================================================
# File: user.py
# Registry: app/models/user.py
# Purpose: User SQLAlchemy model (F01: ログイン機能)
# ======================================================================

from __future__ import annotations

from datetime import datetime
from typing import List

from flask_login import UserMixin
from sqlalchemy import Integer, String, Boolean, DateTime
from werkzeug.security import generate_password_hash, check_password_hash

from app.extensions import db


class User(UserMixin, db.Model):
    __tablename__ = "user"

    id = db.Column(Integer, primary_key=True)
    username = db.Column(String(64), unique=True, nullable=False, index=True)
    password_hash = db.Column(String(256), nullable=False)
    display_name = db.Column(String(128), nullable=True)
    is_admin = db.Column(Boolean, default=True, nullable=False)
    is_active = db.Column(Boolean, default=True, nullable=False)
    created_at = db.Column(DateTime, default=datetime.utcnow)

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username!r}>"
