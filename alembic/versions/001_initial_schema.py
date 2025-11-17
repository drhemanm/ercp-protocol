"""Initial schema

Revision ID: 001_initial
Revises: 
Create Date: 2025-01-17 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create traces table
    op.create_table(
        'traces',
        sa.Column('trace_id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('problem_id', sa.String(255), nullable=True, index=True),
        sa.Column('problem_description', sa.Text(), nullable=False),
        sa.Column('status', sa.String(50), nullable=False, index=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('config', postgresql.JSONB(), nullable=True),
        sa.Column('final_reasoning', sa.Text(), nullable=True),
        sa.Column('utility_score', sa.Float(), nullable=True),
        sa.Column('iteration_count', sa.Integer(), default=0),
        sa.Column('constraint_count', sa.Integer(), default=0),
        sa.Column('error_count', sa.Integer(), default=0),
    )

    # Create trace_events table
    op.create_table(
        'trace_events',
        sa.Column('event_id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('trace_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('operator', sa.String(50), nullable=False, index=True),
        sa.Column('iteration', sa.Integer(), nullable=False, index=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('input_summary', postgresql.JSONB(), nullable=True),
        sa.Column('output_summary', postgresql.JSONB(), nullable=True),
        sa.Column('model_fingerprint', sa.String(255), nullable=True),
        sa.Column('node_signature', sa.String(255), nullable=True),
        sa.Column('duration_seconds', sa.Float(), nullable=True),
        sa.Column('success', sa.Boolean(), default=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['trace_id'], ['traces.trace_id'], ondelete='CASCADE'),
    )
    op.create_index('idx_trace_iteration', 'trace_events', ['trace_id', 'iteration'])
    op.create_index('idx_trace_operator', 'trace_events', ['trace_id', 'operator'])

    # Create constraints table
    op.create_table(
        'constraints',
        sa.Column('constraint_id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('trace_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('type', sa.String(50), nullable=False, index=True),
        sa.Column('priority', sa.Integer(), nullable=False, default=50, index=True),
        sa.Column('nl_text', sa.Text(), nullable=False),
        sa.Column('predicate', postgresql.JSONB(), nullable=True),
        sa.Column('source', postgresql.JSONB(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=False, default=0.0),
        sa.Column('immutable', sa.Boolean(), default=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['trace_id'], ['traces.trace_id'], ondelete='CASCADE'),
    )
    op.create_index('idx_trace_priority', 'constraints', ['trace_id', 'priority'])
    op.create_index('idx_trace_type', 'constraints', ['trace_id', 'type'])

    # Create errors table
    op.create_table(
        'errors',
        sa.Column('error_id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('trace_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('event_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('type', sa.String(50), nullable=False, index=True),
        sa.Column('span', postgresql.JSONB(), nullable=True),
        sa.Column('excerpt', sa.Text(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=False, default=0.0, index=True),
        sa.Column('detected_by', sa.String(50), nullable=True),
        sa.Column('evidence', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['trace_id'], ['traces.trace_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['event_id'], ['trace_events.event_id'], ondelete='CASCADE'),
    )
    op.create_index('idx_trace_confidence', 'errors', ['trace_id', 'confidence'])
    op.create_index('idx_trace_type_error', 'errors', ['trace_id', 'type'])

    # Create model_cache table
    op.create_table(
        'model_cache',
        sa.Column('cache_key', sa.String(255), primary_key=True),
        sa.Column('model_name', sa.String(255), nullable=False, index=True),
        sa.Column('prompt_hash', sa.String(64), nullable=False, index=True),
        sa.Column('output', postgresql.JSONB(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True, index=True),
        sa.Column('hit_count', sa.Integer(), default=0),
        sa.Column('last_accessed', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )
    op.create_index('idx_prompt_model', 'model_cache', ['prompt_hash', 'model_name'])


def downgrade() -> None:
    op.drop_table('model_cache')
    op.drop_table('errors')
    op.drop_table('constraints')
    op.drop_table('trace_events')
    op.drop_table('traces')
