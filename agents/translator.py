from core.state import TranslationState


def translator_node(state: TranslationState) -> dict:
    """
    Iterates over parsed blocks and translates them into the target language,
    respecting the cross-page memory dictionary.
    """
    print("--- [Translator] Starting block-by-block translation ---")
    parsed_data = state.get("parsed_json", {})
    memory = state.get("memory_dict", {})
    blocks = parsed_data.get("blocks", [])

    translated_blocks = []

    # TODO: Implement actual LLM call for translation
    for block in blocks:
        source_text = block["text"]
        # Mock translation process
        target_text = f"[Translated] {source_text}"

        translated_blocks.append({
            "id": block["id"],
            "box": block["box"],
            "source_text": source_text,
            "target_text": target_text
        })

    return {"translated_blocks": translated_blocks}