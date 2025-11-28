"""
Basic tests for ERCP Protocol.
These tests verify core functionality without requiring external services.
"""

import pytest
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestImports:
    """Test that all core modules can be imported."""

    def test_import_middleware_cors(self):
        """Test CORS middleware imports."""
        from server.middleware.cors import get_cors_origins, get_cors_headers, add_cors_middleware
        assert callable(get_cors_origins)
        assert callable(get_cors_headers)
        assert callable(add_cors_middleware)

    def test_import_middleware_sanitization(self):
        """Test sanitization middleware imports."""
        from server.middleware.sanitization import SanitizationMiddleware
        assert SanitizationMiddleware is not None

    def test_import_auth_jwt(self):
        """Test JWT auth imports without requiring JWT_SECRET_KEY."""
        from server.auth.jwt_auth import TokenData, get_secret_key
        assert TokenData is not None
        assert callable(get_secret_key)

    def test_import_database(self):
        """Test database module imports."""
        from server.db.database import get_db, get_db_with_commit, Base
        assert callable(get_db)
        assert callable(get_db_with_commit)
        assert Base is not None


class TestCorsConfiguration:
    """Test CORS configuration."""

    def test_get_cors_headers_returns_list(self):
        """Test that get_cors_headers returns a list of headers."""
        from server.middleware.cors import get_cors_headers
        headers = get_cors_headers()
        assert isinstance(headers, list)
        assert len(headers) > 0

    def test_cors_headers_include_required(self):
        """Test that required headers are included."""
        from server.middleware.cors import get_cors_headers
        headers = get_cors_headers()
        required = ['Authorization', 'Content-Type', 'Accept']
        for h in required:
            assert h in headers, f"Missing required header: {h}"

    def test_get_cors_origins_returns_list(self):
        """Test that get_cors_origins returns a list."""
        from server.middleware.cors import get_cors_origins
        origins = get_cors_origins()
        assert isinstance(origins, list)


class TestJwtAuth:
    """Test JWT authentication module."""

    def test_token_data_model(self):
        """Test TokenData model creation."""
        from server.auth.jwt_auth import TokenData

        # Test with all fields
        td = TokenData(user_id="test_user", roles=["admin", "user"])
        assert td.user_id == "test_user"
        assert "admin" in td.roles
        assert "user" in td.roles

    def test_token_data_default_roles(self):
        """Test TokenData default roles is empty list."""
        from server.auth.jwt_auth import TokenData

        td = TokenData(user_id="test_user")
        assert td.roles == []

    def test_get_secret_key_requires_env_var(self):
        """Test that get_secret_key raises error without JWT_SECRET_KEY."""
        from server.auth.jwt_auth import get_secret_key

        # Clear cache and env var
        get_secret_key.cache_clear()
        original = os.environ.pop('JWT_SECRET_KEY', None)

        try:
            with pytest.raises(RuntimeError) as exc_info:
                get_secret_key()
            assert "JWT_SECRET_KEY" in str(exc_info.value)
        finally:
            # Restore env var if it existed
            if original:
                os.environ['JWT_SECRET_KEY'] = original
            get_secret_key.cache_clear()

    def test_get_secret_key_validates_length(self):
        """Test that get_secret_key validates minimum length."""
        from server.auth.jwt_auth import get_secret_key

        # Clear cache
        get_secret_key.cache_clear()
        original = os.environ.get('JWT_SECRET_KEY')

        try:
            # Set short key
            os.environ['JWT_SECRET_KEY'] = 'tooshort'
            with pytest.raises(RuntimeError) as exc_info:
                get_secret_key()
            assert "32 characters" in str(exc_info.value)
        finally:
            # Restore
            if original:
                os.environ['JWT_SECRET_KEY'] = original
            else:
                os.environ.pop('JWT_SECRET_KEY', None)
            get_secret_key.cache_clear()


class TestSanitization:
    """Test input sanitization."""

    def test_dangerous_patterns_detect_sql_injection(self):
        """Test SQL injection detection."""
        from server.middleware.sanitization import SanitizationMiddleware

        middleware = SanitizationMiddleware(app=None)

        # These should be detected
        dangerous = [
            "'; DROP TABLE users--",
            "1' OR '1'='1",
            "UNION SELECT * FROM users",
            "; DELETE FROM users",
        ]

        for payload in dangerous:
            result = middleware._check_dangerous_patterns(payload)
            assert result is not None, f"Should detect: {payload}"

    def test_safe_content_allowed(self):
        """Test that safe content is allowed."""
        from server.middleware.sanitization import SanitizationMiddleware

        middleware = SanitizationMiddleware(app=None)

        # These should be allowed
        safe = [
            "Hello, world!",
            "This is a normal sentence.",
            "SELECT is a word in English",
            "The price is 10-20 dollars",
        ]

        for payload in safe:
            result = middleware._check_dangerous_patterns(payload)
            assert result is None, f"Should allow: {payload}"


class TestRateLimiting:
    """Test rate limiting configuration."""

    def test_rate_limit_uses_sha256(self):
        """Test that rate limiting uses SHA256 not MD5."""
        with open('server/middleware/rate_limit.py', 'r') as f:
            content = f.read()

        assert 'sha256' in content.lower()
        # MD5 should not be used
        assert 'md5' not in content.lower()


class TestDatabaseConfiguration:
    """Test database configuration."""

    def test_get_db_exists(self):
        """Test get_db function exists."""
        from server.db import get_db
        assert callable(get_db)

    def test_get_db_with_commit_exists(self):
        """Test get_db_with_commit function exists."""
        from server.db import get_db_with_commit
        assert callable(get_db_with_commit)


class TestDockerConfiguration:
    """Test Docker configuration files."""

    def test_dockerfile_uses_python_311(self):
        """Test Dockerfile uses Python 3.11."""
        with open('Dockerfile', 'r') as f:
            content = f.read()
        assert 'python:3.11' in content

    def test_dockerfile_uses_production_server(self):
        """Test Dockerfile uses production server."""
        with open('Dockerfile', 'r') as f:
            content = f.read()
        assert 'ercp_server_v2:app' in content

    def test_dockerfile_has_health_check(self):
        """Test Dockerfile has health check."""
        with open('Dockerfile', 'r') as f:
            content = f.read()
        assert 'HEALTHCHECK' in content

    def test_dockerfile_runs_as_nonroot(self):
        """Test Dockerfile runs as non-root user."""
        with open('Dockerfile', 'r') as f:
            content = f.read()
        assert 'USER ercp' in content or 'USER 1000' in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
