"""Add full-text search index for problem_description

Revision ID: 002_add_fulltext_search
Revises: 001_initial_schema
Create Date: 2025-01-17

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '002_add_fulltext_search'
down_revision = '001_initial_schema'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Add full-text search capabilities to traces.problem_description.

    Creates:
    1. A generated tsvector column for full-text search
    2. A GIN index on the tsvector column for fast full-text queries
    3. A trigger to automatically update the tsvector column
    """
    # Add tsvector column for full-text search
    op.execute("""
        ALTER TABLE traces
        ADD COLUMN problem_description_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('english', problem_description)) STORED;
    """)

    # Create GIN index on tsvector column for fast full-text search
    op.execute("""
        CREATE INDEX idx_traces_problem_description_tsv
        ON traces
        USING GIN (problem_description_tsv);
    """)

    # Optional: Create an additional index for prefix matching (for autocomplete)
    op.execute("""
        CREATE INDEX idx_traces_problem_description_trgm
        ON traces
        USING GIN (problem_description gin_trgm_ops);
    """)

    # Enable pg_trgm extension if not already enabled (for trigram similarity)
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")


def downgrade() -> None:
    """Remove full-text search capabilities."""
    # Drop indexes
    op.execute("DROP INDEX IF EXISTS idx_traces_problem_description_trgm;")
    op.execute("DROP INDEX IF EXISTS idx_traces_problem_description_tsv;")

    # Drop tsvector column
    op.execute("ALTER TABLE traces DROP COLUMN IF EXISTS problem_description_tsv;")
