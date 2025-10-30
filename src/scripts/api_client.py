"""
API client for closed models via DeepInfra and OpenAI
"""

import os
import json
import time
import logging
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def exponential_backoff(
    attempt: int, base_delay: float = 1.0, max_delay: float = 60.0
) -> float:
    """Calculate exponential backoff delay."""
    delay = base_delay * (2**attempt)
    return min(delay, max_delay)


def rate_limit_sleep(last_call_time: float, min_interval: float = 1.0):
    """Ensure minimum interval between API calls."""
    elapsed = time.time() - last_call_time
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)


class DeepInfraClient:
    """Client for DeepInfra API calls."""

    def __init__(
        self,
        base_url: str = "https://api.deepinfra.com/v1",
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()

        # Load environment variables from .env file if available
        if load_dotenv:
            # Look for .env file in the project root (parent directories)
            current_dir = Path(__file__).parent
            env_paths = [
                current_dir / ".env",  # scripts/.env
                current_dir.parent / ".env",  # arabic_accounting_mcq_eval/.env
                current_dir.parent.parent / ".env",  # SAHM_private/.env (root)
            ]

            for env_path in env_paths:
                if env_path.exists():
                    load_dotenv(env_path)
                    logging.info(f"Loaded environment variables from {env_path}")
                    break

        # Get API key from environment
        self.api_key = os.environ.get("DEEPINFRA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DEEPINFRA_API_KEY not found. Please set it in your .env file or environment variables."
            )

        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

        self.last_call_time = 0

        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

        self.last_call_time = 0

    def _make_request(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 32,
    ) -> str:
        """Make a single API request with retry logic."""

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        for attempt in range(self.max_retries + 1):
            try:
                # Rate limiting
                rate_limit_sleep(self.last_call_time, min_interval=0.5)

                response = self.session.post(
                    f"{self.base_url}/openai/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )

                self.last_call_time = time.time()

                response.raise_for_status()
                result = response.json()

                if "choices" not in result or not result["choices"]:
                    raise ValueError(f"No choices in response: {result}")

                content = result["choices"][0]["message"]["content"]
                return content.strip()

            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries:
                    raise Exception(
                        f"API request failed after {self.max_retries + 1} attempts: {e}"
                    )

                delay = exponential_backoff(attempt, self.retry_delay)
                logging.warning(
                    f"API request failed (attempt {attempt + 1}), retrying in {delay:.1f}s: {e}"
                )
                time.sleep(delay)

            except Exception as e:
                if attempt == self.max_retries:
                    raise Exception(
                        f"Unexpected error after {self.max_retries + 1} attempts: {e}"
                    )

                delay = exponential_backoff(attempt, self.retry_delay)
                logging.warning(
                    f"Unexpected error (attempt {attempt + 1}), retrying in {delay:.1f}s: {e}"
                )
                time.sleep(delay)

        raise Exception("Should not reach here")

    def generate(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 32,
    ) -> str:
        """Generate response from closed model."""
        try:
            return self._make_request(model_name, messages, temperature, max_tokens)
        except Exception as e:
            logging.error(f"Failed to generate response with {model_name}: {e}")
            return ""


class OpenAIClient:
    """Client for OpenAI API calls."""

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()

        # Load environment variables from .env file if available
        if load_dotenv:
            # Look for .env file in the project root (parent directories)
            current_dir = Path(__file__).parent
            env_paths = [
                current_dir / ".env",  # scripts/.env
                current_dir.parent / ".env",  # arabic_accounting_mcq_eval/.env
                current_dir.parent.parent / ".env",  # SAHM_private/.env (root)
            ]

            for env_path in env_paths:
                if env_path.exists():
                    load_dotenv(env_path)
                    logging.info(f"Loaded environment variables from {env_path}")
                    break

        # Get API key from environment
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it in your .env file or environment variables."
            )

        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

        self.last_call_time = 0

    def _make_request(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 32,
    ) -> str:
        """Make a single API request with retry logic."""

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        for attempt in range(self.max_retries + 1):
            try:
                # Rate limiting
                rate_limit_sleep(self.last_call_time, min_interval=0.5)

                response = self.session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )

                self.last_call_time = time.time()

                response.raise_for_status()
                result = response.json()

                if "choices" not in result or not result["choices"]:
                    raise ValueError(f"No choices in response: {result}")

                content = result["choices"][0]["message"]["content"]
                return content.strip()

            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries:
                    raise Exception(
                        f"API request failed after {self.max_retries + 1} attempts: {e}"
                    )

                delay = exponential_backoff(attempt, self.retry_delay)
                logging.warning(
                    f"API request failed (attempt {attempt + 1}), retrying in {delay:.1f}s: {e}"
                )
                time.sleep(delay)

            except Exception as e:
                if attempt == self.max_retries:
                    raise Exception(
                        f"Unexpected error after {self.max_retries + 1} attempts: {e}"
                    )

                delay = exponential_backoff(attempt, self.retry_delay)
                logging.warning(
                    f"Unexpected error (attempt {attempt + 1}), retrying in {delay:.1f}s: {e}"
                )
                time.sleep(delay)

        raise Exception("Should not reach here")

    def generate(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 32,
    ) -> str:
        """Generate response from OpenAI model."""
        try:
            return self._make_request(model_name, messages, temperature, max_tokens)
        except Exception as e:
            logging.error(f"Failed to generate response with {model_name}: {e}")
            return ""


class MultiProviderClient:
    """Client that can handle multiple API providers."""

    def __init__(self, config: Dict[str, Any]):
        self.providers = {}

        # Initialize DeepInfra client if configured
        if "deepinfra" in config.get("api", {}):
            try:
                print("initializing deepinfra client")
                api_config = config["api"]["deepinfra"]
                self.providers["deepinfra"] = DeepInfraClient(
                    base_url=api_config.get("base_url", "https://api.deepinfra.com/v1"),
                    timeout=api_config.get("timeout", 120),
                    max_retries=api_config.get("max_retries", 3),
                    retry_delay=api_config.get("retry_delay", 1.0),
                )
                logging.info("DeepInfra client initialized successfully")
            except Exception as e:
                logging.warning(f"Failed to initialize DeepInfra client: {e}")

        # Initialize OpenAI client if configured
        if "openai" in config.get("api", {}):
            try:
                api_config = config["api"]["openai"]
                self.providers["openai"] = OpenAIClient(
                    base_url=api_config.get("base_url", "https://api.openai.com/v1"),
                    timeout=api_config.get("timeout", 120),
                    max_retries=api_config.get("max_retries", 3),
                    retry_delay=api_config.get("retry_delay", 1.0),
                )
                logging.info("OpenAI client initialized successfully")
            except Exception as e:
                logging.warning(f"Failed to initialize OpenAI client: {e}")

        if not self.providers:
            raise ValueError("No API providers were successfully initialized")

    def generate(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 32,
        provider: str = "deepinfra",
    ) -> str:
        """Generate response using the specified provider."""
        if provider not in self.providers:
            raise ValueError(
                f"Provider {provider} not available. Available providers: {list(self.providers.keys())}"
            )

        return self.providers[provider].generate(
            model_name, messages, temperature, max_tokens
        )


def create_api_client(config: Dict[str, Any]):
    """Create multi-provider API client from configuration."""
    return MultiProviderClient(config)


# Legacy function for backward compatibility
def create_deepinfra_client(config: Dict[str, Any]) -> DeepInfraClient:
    """Create DeepInfra API client from configuration."""
    api_config = config.get("api", {}).get("deepinfra", {})
    return DeepInfraClient(
        base_url=api_config.get("base_url", "https://api.deepinfra.com/v1"),
        timeout=api_config.get("timeout", 120),
        max_retries=api_config.get("max_retries", 3),
        retry_delay=api_config.get("retry_delay", 1.0),
    )
