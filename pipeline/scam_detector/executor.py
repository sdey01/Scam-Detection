from typing import Optional

from config import DEFAULT_MODEL
from llm.client import call as call_llm
from utils import get_logger

logger = get_logger(__name__)


def execute(prompt: str, model: Optional[str] = None) -> str:
    """Execute a prompt using the configured Groq model."""
    logger.info("Executing LLM with prompt length: %s", len(prompt))
    try:
        response = call_llm(prompt, model_name=model or DEFAULT_MODEL)
        logger.info("LLM execution successful, response length: %s", len(response))
        return response
    except Exception:
        logger.exception("LLM execution failed")
        raise
