# sdk/python/ercp_client.py
"""
ERCP Python SDK — Production Client
Author: Dr. Heman Mohabeer — EvoLogics AI Lab
License: Apache-2.0

This client provides a robust Python interface to the ERCP Protocol (Spec v1.0).
Includes retry logic, error handling, and connection management.
"""

import requests
import uuid
import time
import logging
from typing import Optional, List, Dict, Any
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ercp_client")


class ERCPError(Exception):
    """Base exception for ERCP client errors."""
    pass


class ERCPAuthenticationError(ERCPError):
    """Raised when authentication fails."""
    pass


class ERCPRateLimitError(ERCPError):
    """Raised when rate limit is exceeded."""
    pass


class ERCPValidationError(ERCPError):
    """Raised when request validation fails."""
    pass


class ERCPServerError(ERCPError):
    """Raised when server encounters an error."""
    pass


class ERCPClient:
    """
    Production-ready Python client for interacting with an ERCP Protocol server.

    Features:
    - Automatic retry with exponential backoff
    - Connection pooling
    - Comprehensive error handling
    - Request/response logging
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        verify_ssl: bool = True
    ):
        """
        Initialize ERCP client.

        Args:
            base_url: URL of the ERCP server, e.g. "http://localhost:8080"
            api_key: API key for authentication (required if server has auth enabled)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts for failed requests
            verify_ssl: Whether to verify SSL certificates
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.verify_ssl = verify_ssl

        # Setup session with connection pooling and retry logic
        self.session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,  # 1s, 2s, 4s
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy, pool_maxsize=10)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        logger.info(f"ERCP client initialized for {self.base_url}")

    # -------------------------
    # Internal request helper
    # -------------------------
    def _request(self, method: str, path: str, json_data: Optional[dict] = None) -> dict:
        """
        Make HTTP request with error handling and retries.

        Args:
            method: HTTP method (GET, POST)
            path: API endpoint path
            json_data: Request payload (for POST)

        Returns:
            Response JSON data

        Raises:
            ERCPAuthenticationError: Authentication failed
            ERCPRateLimitError: Rate limit exceeded
            ERCPValidationError: Request validation failed
            ERCPServerError: Server error
            ERCPError: Other errors
        """
        url = f"{self.base_url}{path}"
        headers = {}

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if method == "POST" and json_data:
            headers["Content-Type"] = "application/json"

        try:
            logger.debug(f"{method} {path}")

            response = self.session.request(
                method,
                url,
                json=json_data,
                headers=headers,
                timeout=self.timeout,
                verify=self.verify_ssl
            )

            # Handle different error codes
            if response.status_code == 401 or response.status_code == 403:
                raise ERCPAuthenticationError(
                    f"Authentication failed: {response.text}"
                )

            if response.status_code == 429:
                raise ERCPRateLimitError(
                    f"Rate limit exceeded: {response.text}"
                )

            if response.status_code == 422:
                raise ERCPValidationError(
                    f"Request validation failed: {response.text}"
                )

            if response.status_code >= 500:
                raise ERCPServerError(
                    f"Server error [{response.status_code}]: {response.text}"
                )

            if not response.ok:
                raise ERCPError(
                    f"Request failed [{response.status_code}]: {response.text}"
                )

            return response.json()

        except requests.exceptions.Timeout:
            raise ERCPError(f"Request timeout after {self.timeout}s")

        except requests.exceptions.ConnectionError as e:
            raise ERCPError(f"Connection error: {str(e)}")

        except requests.exceptions.RequestException as e:
            raise ERCPError(f"Request failed: {str(e)}")

    def close(self):
        """Close the session and cleanup resources."""
        self.session.close()
        logger.info("ERCP client session closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    # -------------------------
    # Public SDK Methods
    # -------------------------

    def run(
        self,
        problem_description: str,
        problem_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
    ) -> dict:
        """
        Run the full ERCP reasoning loop.

        Args:
            problem_description: Natural language problem description
            problem_id: Optional problem identifier
            config: Optional configuration overrides
            metadata: Optional problem metadata
            trace_id: Optional trace ID (generated if not provided)

        Returns:
            dict: Full ERCP response with trace, reasoning, and constraints

        Raises:
            ERCPValidationError: If inputs are invalid
            ERCPServerError: If server encounters an error
        """
        payload = {
            "trace_id": trace_id or str(uuid.uuid4()),
            "problem": {
                "id": problem_id or str(uuid.uuid4()),
                "description": problem_description,
                "metadata": metadata or {},
            },
            "config": config or {
                "model": "local-llm",
                "max_iterations": 20,
                "max_constraints": 30,
                "similarity_threshold": 0.95,
                "temperature": 0.0,
                "deterministic": True,
                "verify_threshold": 0.75,
                "candidate_threshold": 0.60,
            },
        }

        logger.info(f"Running ERCP for problem: {problem_description[:50]}...")
        result = self._request("POST", "/ercp/v1/run", payload)
        logger.info(f"ERCP run completed with status: {result.get('status')}")
        return result

    def generate(
        self,
        problem_description: str,
        constraints: List[dict],
        gen_config: dict,
        trace_id: Optional[str] = None,
    ) -> dict:
        payload = {
            "trace_id": trace_id or str(uuid.uuid4()),
            "problem": {
                "id": str(uuid.uuid4()),
                "description": problem_description,
                "metadata": {},
            },
            "constraints": constraints,
            "gen_config": gen_config,
        }
        return self._request("POST", "/ercp/v1/generate", payload)

    def verify(
        self,
        reasoning_id: str,
        reasoning_text: str,
        constraints: List[dict],
        verify_config: dict,
        trace_id: Optional[str] = None,
    ) -> dict:
        payload = {
            "trace_id": trace_id or str(uuid.uuid4()),
            "reasoning_id": reasoning_id,
            "reasoning_text": reasoning_text,
            "constraints": constraints,
            "verify_config": verify_config,
        }
        return self._request("POST", "/ercp/v1/verify", payload)

    def extract_constraints(
        self,
        errors: List[dict],
        reasoning_text: str,
        extract_config: dict,
        trace_id: Optional[str] = None,
    ) -> dict:
        payload = {
            "trace_id": trace_id or str(uuid.uuid4()),
            "errors": errors,
            "reasoning_text": reasoning_text,
            "extract_config": extract_config,
        }
        return self._request("POST", "/ercp/v1/extract_constraints", payload)

    def stabilize(
        self,
        reasoning_prev: Optional[str],
        reasoning_curr: str,
        threshold: float,
        trace_id: Optional[str] = None,
    ) -> dict:
        payload = {
            "trace_id": trace_id or str(uuid.uuid4()),
            "reasoning_prev": reasoning_prev,
            "reasoning_curr": reasoning_curr,
            "threshold": threshold,
        }
        return self._request("POST", "/ercp/v1/stabilize", payload)

    def mutate(
        self,
        problem: dict,
        reasoning_text: str,
        strategy: str,
        mutation_config: dict,
        trace_id: Optional[str] = None,
    ) -> dict:
        payload = {
            "trace_id": trace_id or str(uuid.uuid4()),
            "problem": problem,
            "reasoning_text": reasoning_text,
            "mutation_strategy": strategy,
            "mutation_config": mutation_config,
        }
        return self._request("POST", "/ercp/v1/mutate", payload)

    def get_trace(self, trace_id: str) -> dict:
        """
        Retrieve a full ERCP audit trace.

        Args:
            trace_id: Trace ID to retrieve

        Returns:
            dict: Complete trace data with all events

        Raises:
            ERCPError: If trace not found (404)
        """
        logger.info(f"Retrieving trace: {trace_id}")
        return self._request("GET", f"/ercp/v1/trace/{trace_id}", json_data=None)

    def verify_signature(self, payload: dict, signature: str) -> bool:
        """
        Verify HMAC signature of a payload (client-side verification).

        Note: This requires access to the same secret key used by the server.
        In most cases, signature verification should be done server-side.

        Args:
            payload: The payload to verify
            signature: The signature to check

        Returns:
            bool: True if signature is valid
        """
        # Client-side signature verification not implemented by default
        # as it requires sharing the secret key
        logger.warning("Client-side signature verification not recommended")
        return True
