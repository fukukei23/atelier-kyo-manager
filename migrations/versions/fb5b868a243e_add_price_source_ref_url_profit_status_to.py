"""add price_source/ref_url/profit_status columns to order

ISSUE-101/102 Phase2 T5: Order に価格出所（α source select）・参照URL（β）・
利益確度（profit_status: confirmed/estimated・estimated は発注ブロック）を追加。

Revision ID: fb5b868a243e
Revises: 208b2c96dcc9
Create Date: 2026-08-14 23:20:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "fb5b868a243e"
down_revision = "208b2c96dcc9"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "order",
        sa.Column("purchase_price_source", sa.String(32), nullable=False, server_default="manual_input"),
    )
    op.add_column(
        "order",
        sa.Column("selling_price_source", sa.String(32), nullable=False, server_default="manual_input"),
    )
    op.add_column("order", sa.Column("purchase_price_ref_url", sa.String(512), nullable=True))
    op.add_column("order", sa.Column("selling_price_ref_url", sa.String(512), nullable=True))
    op.add_column(
        "order",
        sa.Column("profit_status", sa.String(16), nullable=False, server_default="confirmed"),
    )
    op.create_index("ix_order_profit_status", "order", ["profit_status"])


def downgrade():
    op.drop_index("ix_order_profit_status", table_name="order")
    op.drop_column("order", "profit_status")
    op.drop_column("order", "selling_price_ref_url")
    op.drop_column("order", "purchase_price_ref_url")
    op.drop_column("order", "selling_price_source")
    op.drop_column("order", "purchase_price_source")
