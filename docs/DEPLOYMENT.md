# ERCP Protocol - Deployment Guide

## Overview

This guide covers deploying the ERCP Protocol server to production environments.

## Quick Start (Local Development)

### Using Docker Compose

1. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env with your configuration
```

2. **Start all services**:
```bash
docker-compose up -d
```

3. **Check health**:
```bash
curl http://localhost:8080/health
```

4. **View logs**:
```bash
docker-compose logs -f ercp-server
```

## Production Deployment

### Prerequisites

- Python 3.10+
- PostgreSQL 15+
- Redis 7+
- 4GB+ RAM (for ML models)
- 2+ CPU cores

### Option 1: Docker Deployment

**Build the image**:
```bash
docker build -t ercp-server:latest .
```

**Run with environment variables**:
```bash
docker run -d \
  -p 8080:8080 \
  -e DATABASE_URL="postgresql+asyncpg://..." \
  -e JWT_SECRET_KEY="your-secret-key" \
  --name ercp-server \
  ercp-server:latest
```

### Option 2: Kubernetes Deployment

**1. Create secrets**:
```bash
# Copy and edit secrets template
cp k8s/secrets.yaml.example k8s/secrets.yaml
# Edit with your actual values

# Apply secrets
kubectl apply -f k8s/secrets.yaml
```

**2. Deploy application**:
```bash
kubectl apply -f k8s/deployment.yaml
```

**3. Check deployment**:
```bash
kubectl get pods -l app=ercp-server
kubectl logs -f deployment/ercp-server
```

**4. Access service**:
```bash
kubectl get service ercp-server
# Use the external IP
```

### Option 3: Cloud Run (GCP)

**Build and push**:
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/ercp-server

gcloud run deploy ercp-server \
  --image gcr.io/YOUR_PROJECT_ID/ercp-server \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars DATABASE_URL="...",JWT_SECRET_KEY="..."
```

## Database Setup

### Initialize Database

**Using Docker Compose**:
Database is automatically initialized on first run.

**Manual setup**:
```bash
# Create database
createdb ercp

# Run migrations
alembic upgrade head
```

### Migrations

**Create a new migration**:
```bash
alembic revision --autogenerate -m "Description of changes"
```

**Apply migrations**:
```bash
alembic upgrade head
```

**Rollback**:
```bash
alembic downgrade -1
```

## Environment Configuration

### Required Variables

```bash
# Security (MUST CHANGE IN PRODUCTION)
JWT_SECRET_KEY=your-secret-key-here
APP_SECRET_KEY=your-app-secret-for-hmac

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/ercp

# Redis
REDIS_URL=redis://localhost:6379/0
```

### Optional Variables

```bash
# Application
ENVIRONMENT=production
LOG_LEVEL=INFO
LOG_FORMAT=json

# ML Models
DEVICE=cpu  # or cuda, cuda:0, etc.
GENERATE_MODEL_NAME=gpt2

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000

# Monitoring
ENABLE_METRICS=true
```

## Monitoring

### Prometheus Metrics

Metrics are exposed at `/metrics`:

```bash
curl http://localhost:8080/metrics
```

**Key metrics**:
- `ercp_runs_total` - Total ERCP runs by status
- `ercp_iteration_count` - Iterations per run
- `ercp_duration_seconds` - Run duration
- `ercp_operator_duration_seconds` - Operator timing

### Logging

Structured JSON logs are written to stdout:

```bash
# View logs
docker-compose logs -f ercp-server

# Or in Kubernetes
kubectl logs -f deployment/ercp-server
```

**Log levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL

## Scaling

### Horizontal Scaling

The server is stateless and can be scaled horizontally:

**Docker Compose**:
```bash
docker-compose up -d --scale ercp-server=3
```

**Kubernetes**:
```bash
kubectl scale deployment ercp-server --replicas=5
```

**Auto-scaling** (Kubernetes HPA is configured in k8s/deployment.yaml):
- Min: 2 replicas
- Max: 10 replicas
- CPU target: 70%
- Memory target: 80%

### Performance Tuning

**Workers**: Adjust uvicorn workers in Dockerfile:
```dockerfile
CMD ["uvicorn", "server.ercp_server_v2:app", "--workers", "4"]
```

**Database pool**:
```bash
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=10
```

**Model caching**: Models are cached in memory after first load.

## Security Checklist

- [ ] Changed `JWT_SECRET_KEY` from default
- [ ] Changed `APP_SECRET_KEY` from default
- [ ] Database uses strong password
- [ ] Redis requires authentication (if exposed)
- [ ] HTTPS/TLS enabled (use reverse proxy)
- [ ] Rate limiting configured
- [ ] CORS origins restricted
- [ ] Firewall rules configured
- [ ] Container runs as non-root user

## Health Checks

**Liveness probe**:
```bash
curl http://localhost:8080/health
```

**Expected response**:
```json
{
  "status": "healthy",
  "version": "2.0",
  "environment": "production"
}
```

## Troubleshooting

### Common Issues

**1. ML models not loading**
```bash
# Check logs for download errors
docker-compose logs ercp-server | grep "Loading"

# Manually download models
docker-compose exec ercp-server python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

**2. Database connection errors**
```bash
# Test database connectivity
docker-compose exec ercp-server python -c "import asyncio; from server.db import engine; asyncio.run(engine.connect())"
```

**3. Out of memory**
```bash
# Increase container memory limits
# In docker-compose.yml:
deploy:
  resources:
    limits:
      memory: 6G
```

## Backup & Recovery

### Database Backup

**Automated backups** (add to cron):
```bash
#!/bin/bash
pg_dump -h localhost -U ercp_user ercp > backup_$(date +%Y%m%d).sql
```

**Restore**:
```bash
psql -h localhost -U ercp_user ercp < backup_20231201.sql
```

### Model Cache

Models are stored in `/app/models/cache`. Volume mount for persistence:

```yaml
volumes:
  - model_cache:/app/models/cache
```

## Support

For issues or questions:
- GitHub Issues: https://github.com/yourusername/ercp-protocol/issues
- Email: team@evologics.ai
