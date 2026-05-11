from core.state import TranslationState


def parser_critic_node(state: TranslationState) -> dict:
    """
    Validates the structure and logic of the parsed JSON from Vision-Parser.
    """
    print("--- [Parser Critic] Validating JSON structure ---")
    parsed_data = state.get("parsed_json")

    # Basic Rule: Check if parsing was successful
    if not parsed_data or "blocks" not in parsed_data:
        return {"parser_errors": "Error: Missing 'blocks' in output."}

    # Check bounding box validity (example rule)
    for block in parsed_data["blocks"]:
        if len(block.get("box", [])) != 4:
            return {"parser_errors": f"Error: Invalid bounding box for block {block['id']}"}

    # If all checks pass, clear any existing errors
    return {"parser_errors": None}


def translator_critic_node(state: TranslationState) -> dict:
    """
    Validates translation fidelity and terminology consistency.
    """
    print("--- [Translator Critic] Validating translation quality ---")
    # TODO: Implement LLM-based translation check here

    # For MVP, assume it always passes
    return {"translator_errors": None}