# ERCP Protocol — Deployment Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Production Deployment](#production-deployment)
3. [Security Best Practices](#security-best-practices)
4. [Scaling & Performance](#scaling--performance)
5. [Monitoring & Logging](#monitoring--logging)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/ercp-protocol.git
   cd ercp-protocol
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env and set ERCP_SECRET_KEY (minimum 32 bytes)
   python -c "import secrets; print('ERCP_SECRET_KEY=' + secrets.token_hex(32))" >> .env
   ```

4. **Run the server:**
   ```bash
   cd server
   uvicorn ercp_server:app --host 0.0.0.0 --port 8080
   ```

5. **Test the server:**
   ```bash
   curl http://localhost:8080/health
   ```

---

## Production Deployment

### Prerequisites

- Python 3.8+
- PostgreSQL (for trace persistence) or Redis (for caching)
- Nginx or similar reverse proxy (for SSL termination)
- Docker (optional but recommended)

### Docker Deployment

1. **Create Dockerfile:**
   ```dockerfile
   FROM python:3.10-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY server/ ./server/
   COPY .env .

   EXPOSE 8080

   CMD ["uvicorn", "server.ercp_server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
   ```

2. **Build and run:**
   ```bash
   docker build -t ercp-server .
   docker run -d -p 8080:8080 --env-file .env ercp-server
   ```

### Using Docker Compose

```yaml
version: '3.8'

services:
  ercp-server:
    build: .
    ports:
      - "8080:8080"
    environment:
      - ERCP_SECRET_KEY=${ERCP_SECRET_KEY}
      - ERCP_API_KEYS=${ERCP_API_KEYS}
      - ENVIRONMENT=production
      - ALLOWED_HOSTS=your-domain.com
    restart: always

  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - ercp-server
```

### Nginx Configuration

```nginx
upstream ercp {
    server ercp-server:8080;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000" always;

    location / {
        proxy_pass http://ercp;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts for long-running requests
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req zone=api_limit burst=20 nodelay;
}
```

---

## Security Best Practices

### 1. Secret Management

**CRITICAL: Never commit secrets to version control**

- Generate strong secret keys:
  ```bash
  python -c "import secrets; print(secrets.token_hex(32))"
  ```

- Use environment variables or secret managers (AWS Secrets Manager, HashiCorp Vault)

- Rotate secrets periodically

### 2. API Key Authentication

- Generate unique API keys for each client:
  ```bash
  python -c "import secrets; print(secrets.token_urlsafe(32))"
  ```

- Store API keys securely (hashed in database)

- Implement key rotation policy

- Use different keys for development/staging/production

### 3. HTTPS/TLS

**REQUIRED in production**

- Use Let's Encrypt for free SSL certificates
- Configure strong cipher suites
- Enable HTTP Strict Transport Security (HSTS)
- Disable TLS 1.0 and 1.1

### 4. Rate Limiting

Configure appropriate rate limits based on your use case:

```bash
# In .env
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
```

For production, use Redis-based rate limiting:
```python
# Replace in-memory rate limiter with Redis
import redis
redis_client = redis.Redis.from_url(os.getenv("REDIS_URL"))
```

### 5. Input Validation

All inputs are validated via Pydantic models:
- Maximum lengths enforced
- Type checking
- Range validation
- PII redaction (basic implementation - enhance for production)

### 6. Network Security

- Use firewall rules to restrict access
- Implement IP whitelisting if possible
- Use VPC/private networks for internal services
- Enable DDoS protection (CloudFlare, AWS Shield)

### 7. Audit Logging

All requests are logged with:
- Timestamp (ISO8601)
- Client ID
- Operation type
- Trace IDs
- HMAC signatures

**Store logs securely and monitor for suspicious activity**

---

## Scaling & Performance

### Horizontal Scaling

1. **Deploy multiple instances:**
   ```bash
   docker-compose up --scale ercp-server=4
   ```

2. **Use load balancer:**
   - Nginx (shown above)
   - AWS Application Load Balancer
   - GCP Load Balancer

### Database for Trace Storage

Replace in-memory storage with PostgreSQL:

```python
# In ercp_server.py
import psycopg2
from psycopg2.pool import SimpleConnectionPool

# Initialize connection pool
db_pool = SimpleConnectionPool(
    1, 20,
    os.getenv("DATABASE_URL")
)

def store_trace(trace_id: str, trace_data: dict):
    conn = db_pool.getconn()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO traces (trace_id, data, created_at) VALUES (%s, %s, NOW())",
            (trace_id, json.dumps(trace_data))
        )
        conn.commit()
    finally:
        db_pool.putconn(conn)
```

### Caching

Implement caching for frequently accessed data:

```python
import redis

redis_client = redis.Redis.from_url(os.getenv("REDIS_URL"))

def get_trace(trace_id: str):
    # Try cache first
    cached = redis_client.get(f"trace:{trace_id}")
    if cached:
        return json.loads(cached)

    # Fallback to database
    trace = fetch_from_db(trace_id)
    redis_client.setex(f"trace:{trace_id}", 3600, json.dumps(trace))
    return trace
```

### Async Processing

For better performance with I/O-bound operations:

```python
# Already implemented - all endpoints use async def
@app.post("/ercp/v1/run")
async def run_ercp(...):
    # Async implementation allows concurrent request handling
    pass
```

### Performance Monitoring

- Monitor response times
- Track error rates
- Monitor resource usage (CPU, memory, disk)
- Set up alerts for anomalies

---

## Monitoring & Logging

### Application Logs

Logs are written to stdout/stderr and include:
- Timestamp
- Log level (INFO, WARNING, ERROR)
- Component name
- Message

**Centralize logs using:**
- ELK Stack (Elasticsearch, Logstash, Kibana)
- CloudWatch Logs (AWS)
- Stackdriver (GCP)
- Datadog, Splunk, etc.

### Metrics

Recommended metrics to track:
- Request rate (requests/second)
- Error rate (errors/total requests)
- Response time (p50, p95, p99)
- Convergence rate (successful/total runs)
- Average iterations per run
- Constraint extraction rate

### Health Checks

The `/health` endpoint provides basic health status:

```bash
curl https://your-domain.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "ercp-1.0",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### Alerting

Set up alerts for:
- High error rates (>5%)
- Slow response times (>30s for /run)
- High rate limit rejections
- Server errors (5xx responses)
- Authentication failures

---

## Troubleshooting

### Common Issues

**1. "ERCP_SECRET_KEY must be at least 32 bytes"**
   - Generate a proper secret key:
     ```bash
     python -c "import secrets; print(secrets.token_hex(32))"
     ```

**2. Authentication failures (401/403)**
   - Verify API key is correct
   - Check that `ERCP_API_KEYS` is set in .env
   - Ensure Bearer token is included in Authorization header

**3. Rate limit errors (429)**
   - Reduce request rate
   - Increase `RATE_LIMIT_REQUESTS` in .env
   - Implement client-side retry with exponential backoff

**4. Validation errors (422)**
   - Check request payload structure
   - Ensure all required fields are present
   - Verify data types and value ranges

**5. Server errors (500)**
   - Check application logs
   - Verify all dependencies are installed
   - Check database/Redis connectivity

**6. Slow performance**
   - Enable caching (Redis)
   - Scale horizontally (more instances)
   - Optimize database queries
   - Use async implementations for I/O

### Debug Mode

For development, enable debug logging:

```bash
# In .env
LOG_LEVEL=DEBUG
```

### Verify Deployment

Run the test suite:

```bash
# Set test environment
export ERCP_SERVER=https://your-domain.com
export TEST_API_KEY=your-test-key

# Run tests
pytest tests/test_comprehensive.py -v
```

---

## Production Checklist

Before going to production:

- [ ] HTTPS/TLS enabled
- [ ] Strong secret key configured (32+ bytes)
- [ ] API keys generated and distributed securely
- [ ] Rate limiting configured
- [ ] CORS origins restricted (not "*")
- [ ] Trusted hosts configured
- [ ] Database for trace persistence
- [ ] Redis for caching and rate limiting
- [ ] Logging centralized
- [ ] Monitoring and alerting set up
- [ ] Health checks configured
- [ ] Load balancer deployed
- [ ] Backups configured
- [ ] Disaster recovery plan documented
- [ ] Security audit completed
- [ ] Performance testing completed
- [ ] Documentation updated

---

## Support

For issues and questions:
- GitHub Issues: https://github.com/your-org/ercp-protocol/issues
- Documentation: See `ERCPSpec.md`
- Theory Paper: `docs/ERCP_v2_Theory_Paper_FINAL.pdf`

---

**Version:** 1.0
**Last Updated:** 2025-01-15
**Author:** Dr. Heman Mohabeer — EvoLogics AI Lab
