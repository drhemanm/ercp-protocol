# ERCP Protocol — Security Guide

## Security Overview

The ERCP Protocol implementation includes multiple layers of security to ensure safe, auditable, and production-ready operation. This document outlines the security features, best practices, and compliance considerations.

---

## Table of Contents

1. [Security Features](#security-features)
2. [Authentication & Authorization](#authentication--authorization)
3. [Data Protection](#data-protection)
4. [Network Security](#network-security)
5. [Auditability & Compliance](#auditability--compliance)
6. [Threat Model](#threat-model)
7. [Incident Response](#incident-response)
8. [Security Checklist](#security-checklist)

---

## Security Features

### Implemented Security Controls

#### 1. **Authentication**
- Bearer token authentication (JWT/API keys)
- Configurable API key management
- Optional anonymous access (disabled by default in production)

#### 2. **Rate Limiting**
- Per-client rate limiting
- Configurable request limits and time windows
- Protection against DoS attacks
- Supports Redis-based distributed rate limiting

#### 3. **Input Validation**
- Strict Pydantic schema validation
- Maximum length enforcement
- Type checking and range validation
- Sanitization of user inputs
- Prevention of injection attacks

#### 4. **Cryptographic Signatures**
- HMAC-SHA256 signing of all responses
- Tamper detection
- Audit trail integrity
- Non-repudiation support

#### 5. **Secure Defaults**
- HTTPS enforcement in production
- Secure secret key requirements (minimum 32 bytes)
- Constant-time signature comparison
- Safe error messages (no information leakage)

#### 6. **CORS Protection**
- Configurable allowed origins
- Credential support
- Restricted methods (GET, POST only)

#### 7. **Trusted Host Middleware**
- Host header validation in production
- Protection against host header injection

---

## Authentication & Authorization

### API Key Authentication

API keys are used to authenticate clients and enforce rate limiting.

#### Generating API Keys

```bash
# Generate a secure API key
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

#### Configuring API Keys

Add to `.env`:
```bash
ERCP_API_KEYS=key1,key2,key3
```

#### Using API Keys (Client)

```python
from ercp_client import ERCPClient

client = ERCPClient(
    base_url="https://api.example.com",
    api_key="your-api-key-here"
)

result = client.run("Your problem description")
```

#### API Key Best Practices

1. **Generate strong keys**: Minimum 32 bytes of random data
2. **Never commit keys**: Use environment variables or secret managers
3. **Rotate regularly**: Implement key rotation policy (e.g., every 90 days)
4. **Unique per client**: Each client should have their own key
5. **Monitor usage**: Track which keys are being used and when
6. **Revoke compromised keys**: Remove from `ERCP_API_KEYS` immediately

### Role-Based Access Control (RBAC)

For advanced use cases, implement RBAC:

```python
# Example RBAC implementation (not included by default)
from enum import Enum

class Role(Enum):
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"

ROLE_PERMISSIONS = {
    Role.ADMIN: ["run", "generate", "verify", "extract", "mutate", "trace"],
    Role.USER: ["run", "generate", "verify", "trace"],
    Role.READONLY: ["trace"]
}

def check_permission(role: Role, operation: str) -> bool:
    return operation in ROLE_PERMISSIONS.get(role, [])
```

---

## Data Protection

### Secret Management

#### Secret Key Requirements

- **Minimum length**: 32 bytes (256 bits)
- **Randomness**: Use cryptographically secure random generator
- **Storage**: Environment variables, not in code
- **Rotation**: Change periodically and after suspected compromise

#### Generating Secret Keys

```bash
# Generate ERCP_SECRET_KEY
python -c "import secrets; print(secrets.token_hex(32))"
```

#### Using Secret Managers

**AWS Secrets Manager:**
```python
import boto3
import os

def get_secret():
    session = boto3.session.Session()
    client = session.client('secretsmanager')
    response = client.get_secret_value(SecretId='ercp-secret-key')
    return response['SecretString']

os.environ['ERCP_SECRET_KEY'] = get_secret()
```

**HashiCorp Vault:**
```python
import hvac

client = hvac.Client(url='https://vault.example.com')
secret = client.secrets.kv.v2.read_secret_version(path='ercp/secret-key')
os.environ['ERCP_SECRET_KEY'] = secret['data']['data']['key']
```

### PII Protection

Basic PII redaction is implemented in `sanitize_text()` function.

**For production, enhance with:**
- Named Entity Recognition (NER) for PII detection
- Regular expressions for common PII patterns
- Tokenization for sensitive data
- Encryption at rest for stored traces

```python
import re

def sanitize_text_production(text: str) -> str:
    """Enhanced PII redaction for production."""
    # Email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)

    # Phone numbers (US format)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)

    # SSN (US format)
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)

    # Credit card numbers
    text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD]', text)

    return text
```

### Encryption

#### In Transit
- **HTTPS/TLS 1.2+** required in production
- Strong cipher suites
- Perfect Forward Secrecy (PFS)

#### At Rest
- Encrypt database at rest
- Encrypt backups
- Secure key storage

```python
from cryptography.fernet import Fernet

# Encrypt trace data before storage
cipher = Fernet(os.getenv('ENCRYPTION_KEY'))
encrypted_trace = cipher.encrypt(json.dumps(trace_data).encode())
```

---

## Network Security

### HTTPS/TLS Configuration

Nginx SSL configuration:
```nginx
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256';
ssl_prefer_server_ciphers on;
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 10m;
ssl_stapling on;
ssl_stapling_verify on;
```

### Firewall Rules

Restrict access to ERCP server:
```bash
# Allow only specific IPs (example using iptables)
iptables -A INPUT -p tcp --dport 8080 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p tcp --dport 8080 -j DROP
```

### DDoS Protection

1. **Rate limiting** (built-in)
2. **CloudFlare** or similar CDN/WAF
3. **AWS Shield** or GCP Cloud Armor
4. **IP reputation filtering**

### Network Isolation

Deploy in private VPC/subnet:
```
[Internet] → [Load Balancer] → [ERCP Server (private subnet)] → [Database (private subnet)]
```

---

## Auditability & Compliance

### Audit Logging

All requests generate audit logs with:
- **Trace ID**: Unique identifier for each run
- **Timestamp**: ISO8601 format
- **Client ID**: Authenticated user/API key
- **Operation**: Type of operation performed
- **Node Signature**: HMAC-SHA256 signature
- **Model Fingerprint**: Model version used

### Trace Storage

Traces are append-only and include:
- Complete event history
- All operator inputs/outputs
- Timing information
- Model configurations
- Signatures for tamper detection

### Compliance Considerations

#### GDPR
- Implement right to erasure (delete traces)
- Data minimization (collect only necessary data)
- PII redaction
- Data retention policies

#### SOC 2
- Access controls (authentication)
- Audit logging (trace events)
- Encryption in transit and at rest
- Monitoring and alerting

#### HIPAA (if handling health data)
- Enhanced encryption
- Access logging
- Data integrity checks
- Business Associate Agreements (BAAs)

---

## Threat Model

### Threats Addressed

| Threat | Mitigation |
|--------|-----------|
| **Unauthorized access** | API key authentication, rate limiting |
| **Data tampering** | HMAC signatures, append-only logs |
| **Injection attacks** | Input validation, parameterized queries |
| **DoS attacks** | Rate limiting, request size limits |
| **MITM attacks** | HTTPS/TLS, HSTS headers |
| **Information disclosure** | Safe error messages, PII redaction |
| **Session hijacking** | Stateless authentication, short-lived tokens |
| **Replay attacks** | Timestamp validation, nonce support |

### Threats Not Fully Addressed

(Require additional implementation)

| Threat | Recommendation |
|--------|----------------|
| **Advanced persistent threats (APT)** | Implement intrusion detection system (IDS) |
| **Insider threats** | Enhanced RBAC, audit all admin actions |
| **Supply chain attacks** | Dependency scanning, signed packages |
| **Zero-day exploits** | Regular updates, security patches |
| **Cryptographic weaknesses** | Regular security audits, algorithm updates |

---

## Incident Response

### Incident Response Plan

1. **Detection**
   - Monitor logs for anomalies
   - Set up alerts for suspicious activity
   - Regular security audits

2. **Analysis**
   - Review audit logs
   - Identify affected systems
   - Determine scope of incident

3. **Containment**
   - Revoke compromised API keys
   - Block malicious IPs
   - Isolate affected systems

4. **Eradication**
   - Patch vulnerabilities
   - Remove malware/backdoors
   - Reset credentials

5. **Recovery**
   - Restore from clean backups
   - Monitor for reinfection
   - Gradual service restoration

6. **Lessons Learned**
   - Document incident
   - Update security controls
   - Train team

### Security Contacts

Report security vulnerabilities to:
- Email: security@example.com
- PGP Key: [Public key fingerprint]

### Emergency Procedures

**If secret key is compromised:**
1. Generate new secret key immediately
2. Update all servers with new key
3. Invalidate all existing signatures
4. Notify affected users
5. Conduct security audit

**If API keys are leaked:**
1. Remove compromised keys from `ERCP_API_KEYS`
2. Generate new keys for affected users
3. Monitor for unauthorized usage
4. Review audit logs

---

## Security Checklist

### Pre-Production

- [ ] Generate strong secret key (32+ bytes)
- [ ] Configure API keys
- [ ] Enable HTTPS/TLS
- [ ] Configure CORS with specific origins
- [ ] Set trusted hosts
- [ ] Enable rate limiting
- [ ] Configure logging
- [ ] Set up monitoring and alerts
- [ ] Implement PII redaction
- [ ] Review firewall rules
- [ ] Configure backups
- [ ] Document incident response plan
- [ ] Conduct security audit
- [ ] Penetration testing completed

### Ongoing

- [ ] Rotate API keys every 90 days
- [ ] Review audit logs weekly
- [ ] Update dependencies monthly
- [ ] Security patches applied within 48 hours
- [ ] Monitor for anomalies
- [ ] Review and update firewall rules
- [ ] Test backups monthly
- [ ] Update incident response plan
- [ ] Security training for team
- [ ] Annual security audit

---

## Security Resources

### Tools

- **OWASP ZAP**: Web application security scanner
- **Bandit**: Python security linter
- **Safety**: Python dependency vulnerability checker
- **Trivy**: Container vulnerability scanner

### Commands

```bash
# Check for known vulnerabilities in dependencies
pip install safety
safety check

# Scan code for security issues
pip install bandit
bandit -r server/

# Scan Docker image
trivy image ercp-server:latest
```

### References

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)

---

**Version:** 1.0
**Last Updated:** 2025-01-15
**Classification:** Public
**Author:** Dr. Heman Mohabeer — EvoLogics AI Lab
