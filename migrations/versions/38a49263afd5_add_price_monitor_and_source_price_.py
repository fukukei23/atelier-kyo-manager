"""add price_monitor and source_price_history

Revision ID: 38a49263afd5
Revises: 446f3af8c029
Create Date: 2026-06-11

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '38a49263afd5'
down_revision = '446f3af8c029'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "price_monitor",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("brand_price_id", sa.Integer(), sa.ForeignKey("brand_price.id"), nullable=True),
        sa.Column("source_url", sa.String(1024), nullable=False),
        sa.Column("brand", sa.String(128), nullable=False),
        sa.Column("product_name", sa.String(255), nullable=False),
        sa.Column("category", sa.String(32), nullable=False, server_default="accessory"),
        sa.Column("currency", sa.String(8), nullable=False, server_default="USD"),
        sa.Column("current_source_price", sa.Float(), nullable=True),
        sa.Column("current_source_price_jpy", sa.Float(), nullable=True),
        sa.Column("buyma_selling_price", sa.Float(), nullable=False),
        sa.Column("is_profitable", sa.Boolean(), server_default="1"),
        sa.Column("latest_profit", sa.Float(), nullable=True),
        sa.Column("latest_profit_rate", sa.Float(), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default="1", nullable=False),
        sa.Column("check_interval_hours", sa.Integer(), server_default="4", nullable=False),
        sa.Column("last_checked_at", sa.DateTime(), nullable=True),
        sa.Column("last_price_changed_at", sa.DateTime(), nullable=True),
        sa.Column("alert_sent_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_price_monitor_active", "price_monitor", ["is_active", "last_checked_at"])

    op.create_table(
        "source_price_history",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("monitor_id", sa.Integer(), sa.ForeignKey("price_monitor.id"), nullable=False),
        sa.Column("source_price", sa.Float(), nullable=False),
        sa.Column("source_price_jpy", sa.Float(), nullable=False),
        sa.Column("exchange_rate", sa.Float(), nullable=False),
        sa.Column("in_stock", sa.Boolean(), server_default="1"),
        sa.Column("is_profitable", sa.Boolean(), server_default="1"),
        sa.Column("profit", sa.Float(), nullable=True),
        sa.Column("profit_rate", sa.Float(), nullable=True),
        sa.Column("price_changed", sa.Boolean(), server_default="0"),
        sa.Column("recorded_at", sa.DateTime(), nullable=True),
    )
    op.create_index(
        "ix_source_price_history_monitor_time",
        "source_price_history",
        ["monitor_id", "recorded_at"],
    )


def downgrade():
    op.drop_index("ix_source_price_history_monitor_time", "source_price_history")
    op.drop_table("source_price_history")
    op.drop_index("ix_price_monitor_active", "price_monitor")
    op.drop_table("price_monitor")
