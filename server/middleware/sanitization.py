"""
Input sanitization middleware for ERCP Protocol.
Prevents injection attacks and malicious inputs with multi-layered defense.
"""

import re
import json
from typing import Dict, Any, Optional, List
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class SanitizationMiddleware(BaseHTTPMiddleware):
    """
    Enhanced middleware to sanitize and validate incoming requests.

    Security layers:
    1. Length limits (prevents DoS and buffer overflow)
    2. Tokenization checks (detects excessive repetition)
    3. Entropy analysis (detects random/obfuscated payloads)
    4. Multi-pattern detection (prompt injection, XSS, code injection)
    5. Structural validation (JSON depth, array size)

    Note: Regex-based detection is inherently limited. For production:
    - Consider using ML-based content moderation (e.g., OpenAI Moderation API)
    - Implement rate limiting per user/IP
    - Add request signing/verification
    - Use WAF (Web Application Firewall) for additional protection
    """

    # Configuration
    MAX_BODY_SIZE = 10 * 1024 * 1024  # 10 MB
    MAX_FIELD_LENGTH = 100_000  # 100KB per field
    MAX_JSON_DEPTH = 10
    MAX_ARRAY_LENGTH = 1000
    MAX_REPEATED_CHARS = 50
    MAX_REPEATED_WORDS = 10

    # Enhanced pattern detection
    # Using character classes and anchors to reduce bypass attempts
    DANGEROUS_PATTERNS = [
        # Prompt injection - multiple variants
        r"(?i)ignore[\s\._-]*(?:all[\s\._-]*)?(?:previous|prior|above|earlier)[\s\._-]*(?:instructions?|prompts?|rules?|commands?)",
        r"(?i)disregard[\s\._-]*(?:all[\s\._-]*)?(?:previous|prior|above|earlier)",
        r"(?i)forget[\s\._-]*(?:everything|all|previous|instructions?)",
        r"(?i)(?:new|updated|revised)[\s\._-]*(?:instructions?|rules?|system[\s\._-]*prompt)",
        r"(?i)you[\s\._-]*are[\s\._-]*now",
        r"(?i)from[\s\._-]*now[\s\._-]*on",
        r"(?i)system[\s\._-]*(?:prompt|message|role)[\s\._-]*:",
        r"(?i)jailbreak|dan[\s\._-]*mode",

        # XSS and HTML injection
        r"(?i)<[\s\._-]*script[\s\._-]*[^>]*>",
        r"(?i)<[\s\._-]*iframe[\s\._-]*[^>]*>",
        r"(?i)<[\s\._-]*object[\s\._-]*[^>]*>",
        r"(?i)<[\s\._-]*embed[\s\._-]*[^>]*>",
        r"(?i)javascript[\s\._-]*:",
        r"(?i)on(?:load|error|click|mouse)[\s\._-]*=",
        r"(?i)data[\s\._-]*:[\s\._-]*text[\s\._-]*/[\s\._-]*html",

        # Code injection
        r"(?i)eval[\s\._-]*\(",
        r"(?i)exec[\s\._-]*\(",
        r"(?i)__import__[\s\._-]*\(",
        r"(?i)subprocess[\s\._-]*\.",
        r"(?i)os[\s\._-]*\.[\s\._-]*system",
        r"(?i)popen[\s\._-]*\(",

        # SQL injection
        r"(?i)'[\s\._-]*(?:or|and)[\s\._-]*'?\d*[\s\._-]*'?[\s\._-]*=[\s\._-]*'?\d*",
        r"(?i);[\s\._-]*drop[\s\._-]*(?:table|database)",
        r"(?i)union[\s\._-]*(?:all[\s\._-]*)?select",
        r"(?i)insert[\s\._-]*into[\s\._-]*\w+[\s\._-]*values",
        # SQL comment injection - must follow a quote or semicolon to reduce false positives
        r"(?i)[';\)]\s*--\s*(?:$|[^\w])",

        # Command injection
        r"(?i)[;&|`$]\s*(?:ls|cat|pwd|whoami|id|uname)",
        r"(?i)\$\([^)]*\)",  # Command substitution
        r"(?i)`[^`]*`",  # Backtick command

        # Path traversal
        r"\.\.[\\/]",
        r"(?i)(?:/etc/passwd|/etc/shadow|c:\\windows)",
    ]

    async def dispatch(self, request: Request, call_next):
        """
        Process request and sanitize input with multi-layered security checks.

        Args:
            request: Incoming request
            call_next: Next middleware/route handler

        Returns:
            Response or error
        """
        # Only check POST/PUT/PATCH requests with body
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                # Read body
                body = await request.body()

                # 1. Check body size (prevent DoS)
                if len(body) > self.MAX_BODY_SIZE:
                    return self._error_response(
                        status_code=413,
                        error="payload_too_large",
                        message=f"Request body exceeds maximum size of {self.MAX_BODY_SIZE} bytes"
                    )

                # Decode body
                try:
                    body_str = body.decode("utf-8")
                except UnicodeDecodeError:
                    return self._error_response(
                        status_code=400,
                        error="invalid_encoding",
                        message="Request body must be valid UTF-8"
                    )

                # 2. Parse JSON if content-type is application/json
                content_type = request.headers.get("content-type", "")
                if "application/json" in content_type:
                    try:
                        body_json = json.loads(body_str)

                        # 3. Validate JSON structure
                        validation_error = self._validate_json_structure(body_json)
                        if validation_error:
                            return validation_error

                        # 4. Check field lengths
                        length_error = self._check_field_lengths(body_json)
                        if length_error:
                            return length_error

                        # 5. Check for repetition attacks
                        repetition_error = self._check_repetition(body_json)
                        if repetition_error:
                            return repetition_error

                    except json.JSONDecodeError:
                        # Not valid JSON, but might be form data
                        pass

                # 6. Pattern-based detection (on full body string)
                pattern_error = self._check_dangerous_patterns(body_str)
                if pattern_error:
                    return pattern_error

                # 7. Check for excessive special characters (possible obfuscation)
                if self._is_likely_obfuscated(body_str):
                    return self._error_response(
                        status_code=400,
                        error="suspicious_input",
                        message="Input contains suspicious patterns"
                    )

            except Exception as e:
                return self._error_response(
                    status_code=400,
                    error="invalid_request",
                    message="Failed to parse request body"
                )

        # Continue processing
        response = await call_next(request)
        return response

    def _error_response(self, status_code: int, error: str, message: str) -> JSONResponse:
        """Create standardized error response."""
        return JSONResponse(
            status_code=status_code,
            content={
                "error": error,
                "message": message,
            },
        )

    def _validate_json_structure(self, data: Any, depth: int = 0) -> Optional[JSONResponse]:
        """
        Validate JSON structure to prevent deeply nested or oversized payloads.

        Args:
            data: JSON data to validate
            depth: Current nesting depth

        Returns:
            Error response if invalid, None if valid
        """
        # Check depth
        if depth > self.MAX_JSON_DEPTH:
            return self._error_response(
                status_code=400,
                error="invalid_structure",
                message=f"JSON nesting depth exceeds maximum of {self.MAX_JSON_DEPTH}"
            )

        # Check arrays
        if isinstance(data, list):
            if len(data) > self.MAX_ARRAY_LENGTH:
                return self._error_response(
                    status_code=400,
                    error="invalid_structure",
                    message=f"Array length exceeds maximum of {self.MAX_ARRAY_LENGTH}"
                )
            for item in data:
                error = self._validate_json_structure(item, depth + 1)
                if error:
                    return error

        # Recursively check dictionaries
        elif isinstance(data, dict):
            for value in data.values():
                error = self._validate_json_structure(value, depth + 1)
                if error:
                    return error

        return None

    def _check_field_lengths(self, data: Any) -> Optional[JSONResponse]:
        """
        Check that all string fields are within reasonable length limits.

        Args:
            data: JSON data to check

        Returns:
            Error response if invalid, None if valid
        """
        if isinstance(data, str):
            if len(data) > self.MAX_FIELD_LENGTH:
                return self._error_response(
                    status_code=400,
                    error="field_too_long",
                    message=f"Field length exceeds maximum of {self.MAX_FIELD_LENGTH} characters"
                )

        elif isinstance(data, list):
            for item in data:
                error = self._check_field_lengths(item)
                if error:
                    return error

        elif isinstance(data, dict):
            for value in data.values():
                error = self._check_field_lengths(value)
                if error:
                    return error

        return None

    def _check_repetition(self, data: Any) -> Optional[JSONResponse]:
        """
        Check for excessive character or word repetition (common in injection attacks).

        Args:
            data: JSON data to check

        Returns:
            Error response if suspicious, None if valid
        """
        if isinstance(data, str):
            # Check for repeated characters (e.g., "aaaaa..." or "=-=-=...")
            char_repetition = re.findall(r"(.)\1{" + str(self.MAX_REPEATED_CHARS - 1) + ",}", data)
            if char_repetition:
                return self._error_response(
                    status_code=400,
                    error="suspicious_repetition",
                    message="Excessive character repetition detected"
                )

            # Check for repeated words
            words = data.lower().split()
            if len(words) > self.MAX_REPEATED_WORDS:
                word_counts = {}
                for word in words:
                    word_counts[word] = word_counts.get(word, 0) + 1
                    if word_counts[word] > self.MAX_REPEATED_WORDS:
                        return self._error_response(
                            status_code=400,
                            error="suspicious_repetition",
                            message="Excessive word repetition detected"
                        )

        elif isinstance(data, list):
            for item in data:
                error = self._check_repetition(item)
                if error:
                    return error

        elif isinstance(data, dict):
            for value in data.values():
                error = self._check_repetition(value)
                if error:
                    return error

        return None

    def _check_dangerous_patterns(self, text: str) -> Optional[JSONResponse]:
        """
        Check for dangerous patterns using regex.

        Args:
            text: Text to check

        Returns:
            Error response if dangerous pattern found, None if safe
        """
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, text):
                return self._error_response(
                    status_code=400,
                    error="malicious_input_detected",
                    message="Input contains potentially malicious patterns"
                )
        return None

    def _is_likely_obfuscated(self, text: str) -> bool:
        """
        Check if text is likely obfuscated (high ratio of special characters).

        Args:
            text: Text to analyze

        Returns:
            True if likely obfuscated, False otherwise
        """
        if len(text) < 20:
            return False

        # Count special characters
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_ratio = special_chars / len(text)

        # If more than 30% special characters, flag as suspicious
        return special_ratio > 0.3
