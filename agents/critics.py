from core.state import TranslationState
from utils.logger import logger


def parser_critic_node(state: TranslationState) -> dict:
    """
    Validates the structure and geometric logic of the parsed JSON.
    """
    logger.info("[Parser Critic] Validating JSON structure and coordinates.")
    parsed_data = state.get("parsed_json")

    if not parsed_data or "blocks" not in parsed_data:
        err = "Error: Missing 'blocks' in output."
        logger.error(f"[Parser Critic] {err}")
        return {"parser_errors": err}

    for block in parsed_data["blocks"]:
        if len(block.get("box", [])) != 4:
            err = f"Error: Invalid bounding box format for block {block['id']}"
            logger.error(f"[Parser Critic] {err}")
            return {"parser_errors": err}

    logger.info("[Parser Critic] Validation passed.")
    return {"parser_errors": None}


def translator_critic_node(state: TranslationState) -> dict:
    """
    Validates translation fidelity, missing blocks, and layout safety (length inflation).
    """
    logger.info("[Translator Critic] Checking translation quality and length constraints.")
    translated_blocks = state.get("translated_blocks", [])
    retry_count = state.get("translator_retry_count", 0)

    if not translated_blocks:
        err = "Critic Error: No translated blocks found."
        logger.error(f"[Translator Critic] {err}")
        return {"translator_errors": err, "translator_retry_count": retry_count + 1}

    errors = []
    for block in translated_blocks:
        t_text = block["target_text"]
        s_text = block["source_text"]
        b_id = block["id"]

        # 1. Missing Translation Check
        if not t_text or "[Warning" in t_text:
            errors.append(f"Block {b_id} is missing translation.")
            continue

        # 2. Length Inflation Check (Heuristic: Chinese is usually much shorter than English)
        if len(t_text) > len(s_text) * 1.5:
            errors.append(
                f"Block {b_id} text is too long (Src: {len(s_text)} chars, Trgt: {len(t_text)} chars). "
                f"This will cause layout overflow. Please condense."
            )

    if errors:
        error_msg = " | ".join(errors)
        logger.warning(f"[Translator Critic] Issues detected: {error_msg}")
        return {"translator_errors": error_msg, "translator_retry_count": retry_count + 1}

    logger.info("[Translator Critic] All translations passed quality and layout checks.")
    return {"translator_errors": None}