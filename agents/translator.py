import json
import os
import re
from openai import OpenAI

from config import PROMPT_DIR
from core.state import TranslationState
from utils.image_utils import encode_image_to_base64
from utils.logger import logger

client = OpenAI()


def load_prompt(filename: str) -> str:
    filepath = os.path.join(PROMPT_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()


def contains_bad_english(text: str) -> bool:
    """Detect obvious untranslated English while allowing names, acronyms, and Latin binomials."""
    cleaned = text
    # Allow content inside parentheses for names/scientific names, e.g. (Daniel Robinson), (Piper methysticum)
    cleaned = re.sub(r"[（(][^）)]*[A-Za-z][^）)]*[）)]", "", cleaned)
    # Allow common acronyms and treaty/organization abbreviations.
    cleaned = re.sub(r"\b(?:UNESCO|WIPO|WTO|CBD|DNA|IP|ABS)\b", "", cleaned)
    # Allow Latin binomial without parentheses.
    cleaned = re.sub(r"\b[A-Z][a-z]+\s+[a-z]+\b", "", cleaned)
    # Allow URLs/emails.
    cleaned = re.sub(r"\S+@\S+|https?://\S+|\w+\.com\b", "", cleaned)
    return bool(re.search(r"[A-Za-z]{3,}", cleaned))


def translator_node(state: TranslationState) -> dict:
    logger.info("[Translator] Starting layout-aware multimodal translation...")

    parsed_json = state.get("parsed_json", {}) or {}
    blocks = parsed_json.get("blocks", [])
    img_path = state.get("image_path")
    memory = state.get("memory_dict", {})
    retry_count = int(state.get("translator_retry_count", 0))
    previous_errors = state.get("translator_errors")

    if not blocks or not img_path:
        return {"translated_blocks": [], "translator_errors": "No parsed blocks or image path.", "translator_retry_count": retry_count + 1}

    source_payload = {}
    for b in blocks:
        bid = str(b.get("id"))
        source_payload[bid] = {
            "text": b.get("text", ""),
            "style": b.get("style", "body"),
            "flow_id": b.get("flow_id", "main"),
            "column": b.get("column", 0),
        }

    prompt_template = load_prompt("vision_translator.txt")
    prompt_text = prompt_template.replace("[MEMORY]", json.dumps(memory, ensure_ascii=False, indent=2)).replace(
        "[SOURCE_TEXTS]", json.dumps(source_payload, ensure_ascii=False, indent=2)
    )
    if previous_errors:
        prompt_text += f"\n\nPrevious critic errors that MUST be fixed:\n{previous_errors}\n"

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_to_base64(img_path)}"}},
                ],
            }],
            temperature=0.2,
            max_tokens=5000,
        )
        translated_dict = json.loads(response.choices[0].message.content or "{}")

        missing = [str(b.get("id")) for b in blocks if str(b.get("id")) not in translated_dict]
        if missing:
            raise ValueError(f"Missing translations for block IDs: {missing}. Fallback to English is forbidden.")

        translated_blocks = []
        residue_errors = []
        for b in blocks:
            bid = str(b.get("id"))
            target_text = str(translated_dict.get(bid, "")).strip()
            if not target_text:
                residue_errors.append(f"Block {bid} is empty.")
            if contains_bad_english(target_text):
                residue_errors.append(f"Block {bid} contains English residue: {target_text[:120]}")

            translated_blocks.append({
                "id": b.get("id"),
                "source_text": b.get("text", ""),
                "target_text": target_text,
                "source_box": b.get("source_box"),
                "erase_boxes": b.get("erase_boxes", []),
                "target_box": b.get("target_box"),
                "style": b.get("style", "body"),
                "column": b.get("column", 0),
                "reading_order": b.get("reading_order", b.get("id", 0)),
                "align": b.get("align", "left"),
                "font_role": b.get("font_role", "body"),
                "color_role": b.get("color_role", "body"),
                "flow_id": b.get("flow_id", "main"),
            })

        logger.info("\n=== [Translator] Output Review ===")
        for b in translated_blocks:
            logger.info(f"Block {b['id']} [{b['style']}] EN={b['source_text'][:45]!r} ZH={b['target_text'][:45]!r}")
        logger.info("================================\n")

        if residue_errors:
            return {
                "translated_blocks": translated_blocks,
                "translator_errors": " | ".join(residue_errors),
                "translator_retry_count": retry_count + 1,
            }

        return {"translated_blocks": translated_blocks, "translator_errors": None, "translator_retry_count": retry_count + 1}

    except Exception as e:
        logger.error(f"[Translator] Failed: {e}", exc_info=True)
        return {"translated_blocks": [], "translator_errors": str(e), "translator_retry_count": retry_count + 1}
