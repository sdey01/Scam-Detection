from typing import Any

from utils import extract_json_from_text, get_logger

logger = get_logger(__name__)


def parse_llm_output(llm_output: str) -> dict[str, Any]:
    """Parse LLM output as JSON, returning an uncertainty result if parsing fails."""
    logger.info("Parsing LLM output of length: %s", len(llm_output))
    parsed_json = extract_json_from_text(llm_output)

    if parsed_json:
        logger.info("Successfully parsed LLM output to JSON.")
        return parsed_json

    logger.warning("No JSON found in LLM output.")
    return {
        "label": "Uncertain",
        "reasoning": "Failed to parse response: No JSON found",
        "intent": "Could not determine",
        "risk_factors": [],
    }
