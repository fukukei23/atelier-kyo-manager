"""add price_source columns to product

Revision ID: 208b2c96dcc9
Revises: 38a49263afd5
Create Date: 2026-07-13 12:20:29.278785

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '208b2c96dcc9'
down_revision = '38a49263afd5'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "product",
        sa.Column("purchase_price_source", sa.String(32), nullable=True, server_default="unknown"),
    )
    op.add_column(
        "product",
        sa.Column("selling_price_source", sa.String(32), nullable=True, server_default="unknown"),
    )


def downgrade():
    op.drop_column("product", "selling_price_source")
    op.drop_column("product", "purchase_price_source")
