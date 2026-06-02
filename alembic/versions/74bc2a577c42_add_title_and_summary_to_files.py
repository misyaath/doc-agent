"""add title and summary to files

Revision ID: 74bc2a577c42
Revises: 2c643dc4b7d0
Create Date: 2026-05-16 06:33:20.229953

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '74bc2a577c42'
down_revision: Union[str, Sequence[str], None] = '2c643dc4b7d0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
