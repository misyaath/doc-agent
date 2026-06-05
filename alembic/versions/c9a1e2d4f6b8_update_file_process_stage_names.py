"""update file process stage names

Revision ID: c9a1e2d4f6b8
Revises: b6d0f2a41c9e
Create Date: 2026-06-04 13:10:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c9a1e2d4f6b8"
down_revision: str | Sequence[str] | None = "b6d0f2a41c9e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_constraint("ck_file_process_stages_stage", "file_process_stages", type_="check")
    op.execute(
        """
        UPDATE file_process_stages
        SET stage = CASE stage
            WHEN 'extracted' THEN 'extracting'
            WHEN 'normalizer' THEN 'analysing'
            WHEN 'enriched' THEN 'organizing'
            WHEN 'embedding' THEN 'saving'
            ELSE stage
        END
        """
    )
    op.create_check_constraint(
        "ck_file_process_stages_stage",
        "file_process_stages",
        "stage IN ('uploaded','extracting','analysing','organizing','summarizing','saving','done')",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("ck_file_process_stages_stage", "file_process_stages", type_="check")
    op.execute(
        """
        DELETE FROM file_process_stages
        WHERE stage = 'summarizing'
        """
    )
    op.execute(
        """
        UPDATE file_process_stages
        SET stage = CASE stage
            WHEN 'extracting' THEN 'extracted'
            WHEN 'analysing' THEN 'normalizer'
            WHEN 'organizing' THEN 'enriched'
            WHEN 'saving' THEN 'embedding'
            ELSE stage
        END
        """
    )
    op.create_check_constraint(
        "ck_file_process_stages_stage",
        "file_process_stages",
        "stage IN ('uploaded','extracted','normalizer','enriched','embedding','done')",
    )
