import time
from typing import Any

from groq import Groq

from config import DEFAULT_MODEL, MAX_RETRIES, RETRY_DELAY
from utils import get_logger

logger = get_logger(__name__)

def call(
    prompt: str,
    model_name: str = DEFAULT_MODEL,
    max_retries: int = MAX_RETRIES,
    retry_delay: float = RETRY_DELAY,
    **kwargs: Any,
) -> str:
    """Send a prompt to the Groq API and return its response text."""
    client = Groq()

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            content = response.choices[0].message.content if response.choices else None
            if content:
                return content.strip()
            raise ValueError("Empty response received")
        except Exception as error:
            if attempt == max_retries:
                raise RuntimeError(
                    f"API call failed after {max_retries + 1} attempts: {error}"
                ) from error

            logger.warning("Groq API call failed; retrying (attempt %s): %s", attempt + 1, error)
            time.sleep(retry_delay * (2**attempt))
