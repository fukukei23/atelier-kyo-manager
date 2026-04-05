#!/usr/bin/env python3
"""
初期管理者ユーザーを作成するスクリプト

使い方:
    python scripts/seed_admin.py [--username USER] [--password PASS]
"""
import sys
from pathlib import Path

# app パスを追加
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import create_app
from app.extensions import db
from app.models.user import User


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Seed initial admin user")
    parser.add_argument("--username", default="admin", help="管理者ユーザー名")
    parser.add_argument("--password", default="changeme", help="管理者パスワード")
    parser.add_argument("--display-name", default="Atelier Kyo", help="表示名")
    args = parser.parse_args()

    app = create_app()
    with app.app_context():
        existing = User.query.filter_by(username=args.username).first()
        if existing:
            print(f"User '{args.username}' already exists (id={existing.id}). Skipping.")
            return

        user = User(
            username=args.username,
            display_name=args.display_name,
            is_admin=True,
            is_active=True,
        )
        user.set_password(args.password)
        db.session.add(user)
        db.session.commit()
        print(f"Created admin user: {args.username} (id={user.id})")


if __name__ == "__main__":
    main()
