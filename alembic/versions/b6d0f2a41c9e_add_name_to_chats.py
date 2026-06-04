"""add name to chats

Revision ID: b6d0f2a41c9e
Revises: e9b62b1a920f
Create Date: 2026-06-04 11:20:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b6d0f2a41c9e"
down_revision: str | Sequence[str] | None = "e9b62b1a920f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "chats",
        sa.Column("name", sa.String(length=120), server_default="New Chat", nullable=False),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("chats", "name")
