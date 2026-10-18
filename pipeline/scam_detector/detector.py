from typing import Any

from .builder import build_prompt
from .executor import execute
from .parser import parse_llm_output
from utils import get_logger

logger = get_logger(__name__)


def detect(message: str, strategy: str = "react") -> dict[str, Any]:
    """Run scam detection on a single message."""
    logger.info("Starting detection for message length: %s", len(message))
    try:
        prompt = build_prompt(message, strategy)
        raw_response = execute(prompt)
        parsed_result = parse_llm_output(raw_response)
        logger.info(
            "Detection successful. Result: %s",
            parsed_result.get("label", "Unknown"),
        )
        return parsed_result
    except Exception:
        logger.exception("Detection pipeline failed")
        raise


def detect_batch(
    messages: list[str], strategy: str = "react"
) -> list[dict[str, Any]]:
    """Run scam detection on a list of messages."""
    total_messages = len(messages)
    logger.info("Starting batch detection for %s messages.", total_messages)
    results = []

    for index, message in enumerate(messages):
        try:
            results.append(detect(message, strategy))
        except Exception as error:
            logger.warning(
                "Failed to process message %s/%s: %s",
                index + 1,
                total_messages,
                error,
            )
            results.append(
                {
                    "label": "Uncertain",
                    "reasoning": f"Error processing message: {error}",
                    "intent": "Could not determine",
                    "risk_factors": ["processing_error"],
                }
            )

    successful = sum(
        1
        for result in results
        if result.get("label") != "Uncertain"
        or "processing_error" not in result.get("risk_factors", [])
    )
    logger.info(
        "Batch processing complete. %s/%s succeeded.", successful, total_messages
    )
    return results
