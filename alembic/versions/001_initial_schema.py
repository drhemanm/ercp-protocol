"""Initial schema with all ERCP tables

Revision ID: 001_initial_schema
Revises:
Create Date: 2025-11-17 09:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all initial tables for ERCP protocol."""

    # Create traces table
    op.create_table(
        'traces',
        sa.Column('trace_id', sa.String(), nullable=False),
        sa.Column('problem_id', sa.String(), nullable=False),
        sa.Column('problem_description', sa.Text(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('config', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('final_reasoning', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('utility_score', sa.Float(), nullable=True),
        sa.Column('iteration_count', sa.Integer(), nullable=True, server_default='0'),
        sa.PrimaryKeyConstraint('trace_id')
    )

    # Create indexes for traces
    op.create_index(op.f('ix_traces_problem_id'), 'traces', ['problem_id'], unique=False)
    op.create_index(op.f('ix_traces_status'), 'traces', ['status'], unique=False)

    # Create trace_events table
    op.create_table(
        'trace_events',
        sa.Column('event_id', sa.String(), nullable=False),
        sa.Column('trace_id', sa.String(), nullable=False),
        sa.Column('operator', sa.String(), nullable=False),
        sa.Column('iteration', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('input_summary', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('output_summary', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('model_fingerprint', sa.String(), nullable=True),
        sa.Column('node_signature', sa.String(), nullable=True),
        sa.Column('duration_ms', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['trace_id'], ['traces.trace_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('event_id')
    )

    # Create indexes for trace_events
    op.create_index(op.f('ix_trace_events_trace_id'), 'trace_events', ['trace_id'], unique=False)
    op.create_index(op.f('ix_trace_events_operator'), 'trace_events', ['operator'], unique=False)
    op.create_index(op.f('ix_trace_events_iteration'), 'trace_events', ['iteration'], unique=False)

    # Create constraints table
    op.create_table(
        'constraints',
        sa.Column('constraint_id', sa.String(), nullable=False),
        sa.Column('trace_id', sa.String(), nullable=False),
        sa.Column('type', sa.String(), nullable=False),
        sa.Column('priority', sa.String(), nullable=False),
        sa.Column('nl_text', sa.Text(), nullable=False),
        sa.Column('predicate', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('source', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('immutable', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['trace_id'], ['traces.trace_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('constraint_id')
    )

    # Create indexes for constraints
    op.create_index(op.f('ix_constraints_trace_id'), 'constraints', ['trace_id'], unique=False)
    op.create_index(op.f('ix_constraints_type'), 'constraints', ['type'], unique=False)

    # Create errors table
    op.create_table(
        'errors',
        sa.Column('error_id', sa.String(), nullable=False),
        sa.Column('trace_id', sa.String(), nullable=False),
        sa.Column('event_id', sa.String(), nullable=True),
        sa.Column('type', sa.String(), nullable=False),
        sa.Column('span', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('excerpt', sa.Text(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('detected_by', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('evidence', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['event_id'], ['trace_events.event_id'], ),
        sa.ForeignKeyConstraint(['trace_id'], ['traces.trace_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('error_id')
    )

    # Create indexes for errors
    op.create_index(op.f('ix_errors_trace_id'), 'errors', ['trace_id'], unique=False)
    op.create_index(op.f('ix_errors_type'), 'errors', ['type'], unique=False)

    # Create model_cache table
    op.create_table(
        'model_cache',
        sa.Column('cache_key', sa.String(), nullable=False),
        sa.Column('model_name', sa.String(), nullable=False),
        sa.Column('prompt_hash', sa.String(), nullable=False),
        sa.Column('output', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('hit_count', sa.Integer(), nullable=True, server_default='0'),
        sa.PrimaryKeyConstraint('cache_key')
    )

    # Create indexes for model_cache
    op.create_index(op.f('ix_model_cache_model_name'), 'model_cache', ['model_name'], unique=False)
    op.create_index(op.f('ix_model_cache_prompt_hash'), 'model_cache', ['prompt_hash'], unique=False)

    # Create composite index for efficient cache lookups
    op.create_index(
        'ix_model_cache_lookup',
        'model_cache',
        ['model_name', 'prompt_hash'],
        unique=False
    )


def downgrade() -> None:
    """Drop all tables (reverse migration)."""

    # Drop tables in reverse order (respecting foreign keys)
    op.drop_index(op.f('ix_model_cache_lookup'), table_name='model_cache')
    op.drop_index(op.f('ix_model_cache_prompt_hash'), table_name='model_cache')
    op.drop_index(op.f('ix_model_cache_model_name'), table_name='model_cache')
    op.drop_table('model_cache')

    op.drop_index(op.f('ix_errors_type'), table_name='errors')
    op.drop_index(op.f('ix_errors_trace_id'), table_name='errors')
    op.drop_table('errors')

    op.drop_index(op.f('ix_constraints_type'), table_name='constraints')
    op.drop_index(op.f('ix_constraints_trace_id'), table_name='constraints')
    op.drop_table('constraints')

    op.drop_index(op.f('ix_trace_events_iteration'), table_name='trace_events')
    op.drop_index(op.f('ix_trace_events_operator'), table_name='trace_events')
    op.drop_index(op.f('ix_trace_events_trace_id'), table_name='trace_events')
    op.drop_table('trace_events')

    op.drop_index(op.f('ix_traces_status'), table_name='traces')
    op.drop_index(op.f('ix_traces_problem_id'), table_name='traces')
    op.drop_table('traces')
