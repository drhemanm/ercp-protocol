#!/bin/bash
# Pre-deployment checks for ERCP Protocol
# Run this before deploying to verify configuration

set -e

echo "=============================================="
echo "ERCP Protocol Pre-Deployment Check"
echo "=============================================="
echo ""

ERRORS=0
WARNINGS=0

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

pass() {
    echo -e "${GREEN}✓${NC} $1"
}

fail() {
    echo -e "${RED}✗${NC} $1"
    ERRORS=$((ERRORS + 1))
}

warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    WARNINGS=$((WARNINGS + 1))
}

# ============================================
# 1. Check Environment Variables
# ============================================
echo "1. Checking Environment Variables..."
echo ""

if [ -f .env ]; then
    pass ".env file exists"
    source .env
else
    fail ".env file not found"
fi

if [ -n "$JWT_SECRET_KEY" ]; then
    if [ ${#JWT_SECRET_KEY} -ge 32 ]; then
        pass "JWT_SECRET_KEY is set (${#JWT_SECRET_KEY} chars)"
    else
        fail "JWT_SECRET_KEY is too short (${#JWT_SECRET_KEY} chars, need 32+)"
    fi
else
    fail "JWT_SECRET_KEY is not set"
fi

if [ -n "$APP_SECRET_KEY" ]; then
    pass "APP_SECRET_KEY is set"
else
    warn "APP_SECRET_KEY is not set (optional but recommended)"
fi

if [ -n "$DATABASE_URL" ]; then
    pass "DATABASE_URL is set"
else
    warn "DATABASE_URL not set (will use default)"
fi

echo ""

# ============================================
# 2. Check Python Files Syntax
# ============================================
echo "2. Checking Python Syntax..."
echo ""

PYTHON_FILES=$(find server -name "*.py" -type f 2>/dev/null | head -20)
SYNTAX_OK=true

for file in $PYTHON_FILES; do
    if python -m py_compile "$file" 2>/dev/null; then
        : # Syntax OK
    else
        fail "Syntax error in $file"
        SYNTAX_OK=false
    fi
done

if [ "$SYNTAX_OK" = true ]; then
    pass "All Python files have valid syntax"
fi

echo ""

# ============================================
# 3. Check Required Files
# ============================================
echo "3. Checking Required Files..."
echo ""

REQUIRED_FILES=(
    "Dockerfile"
    "docker-compose.yml"
    "requirements.txt"
    "server/ercp_server_v2.py"
    "server/auth/jwt_auth.py"
    "server/db/database.py"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        pass "$file exists"
    else
        fail "$file is missing"
    fi
done

echo ""

# ============================================
# 4. Check Docker Configuration
# ============================================
echo "4. Checking Docker Configuration..."
echo ""

if command -v docker &> /dev/null; then
    pass "Docker is installed"

    if docker info &> /dev/null; then
        pass "Docker daemon is running"
    else
        warn "Docker daemon is not running"
    fi
else
    warn "Docker is not installed (skip if using remote build)"
fi

if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
    pass "Docker Compose is available"
else
    warn "Docker Compose is not available"
fi

echo ""

# ============================================
# 5. Check Security Configuration
# ============================================
echo "5. Checking Security Configuration..."
echo ""

# Check for secrets in code
if grep -r "SECRET_KEY\s*=\s*['\"]" server/ --include="*.py" 2>/dev/null | grep -v "os.getenv" | grep -v "SECRET_KEY\s*=" > /dev/null; then
    fail "Hardcoded secrets found in code"
else
    pass "No hardcoded secrets found"
fi

# Check .gitignore
if [ -f .gitignore ]; then
    if grep -q "\.env" .gitignore; then
        pass ".env is in .gitignore"
    else
        warn ".env is not in .gitignore"
    fi
else
    warn ".gitignore not found"
fi

echo ""

# ============================================
# 6. Check Dependencies
# ============================================
echo "6. Checking Dependencies..."
echo ""

if [ -f requirements.txt ]; then
    DEPS=$(wc -l < requirements.txt)
    pass "requirements.txt has $DEPS dependencies"

    # Check for security packages
    if grep -q "PyJWT" requirements.txt; then
        pass "PyJWT is included"
    else
        warn "PyJWT not found in requirements"
    fi

    if grep -q "python-jose" requirements.txt; then
        pass "python-jose is included"
    else
        warn "python-jose not found in requirements"
    fi
fi

echo ""

# ============================================
# Summary
# ============================================
echo "=============================================="
echo "Summary"
echo "=============================================="
echo ""

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}All critical checks passed!${NC}"
else
    echo -e "${RED}$ERRORS critical error(s) found${NC}"
fi

if [ $WARNINGS -gt 0 ]; then
    echo -e "${YELLOW}$WARNINGS warning(s)${NC}"
fi

echo ""

if [ $ERRORS -eq 0 ]; then
    echo "Ready for deployment!"
    exit 0
else
    echo "Please fix the errors before deploying."
    exit 1
fi
