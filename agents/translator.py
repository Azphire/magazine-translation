import os
import json
import time
from typing import List
from pydantic import BaseModel, Field
from core.state import TranslationState
from utils.llm_client import get_gpt4o_client
from utils.logger import logger
from langchain_core.messages import HumanMessage, SystemMessage


def load_prompt(filename: str) -> str:
    """
    Loads prompt content from the 'prompts' directory.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(base_dir, "prompts", filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()


# 1. Define the desired JSON structure for batch translation
class TranslatedBlock(BaseModel):
    id: int = Field(description="The unique ID of the text block (must match the input ID)")
    target_text: str = Field(description="The translated Chinese text")


class BatchTranslationResponse(BaseModel):
    translations: List[TranslatedBlock] = Field(description="List of all translated blocks")


def translator_node(state: TranslationState) -> dict:
    """
    Calls GPT-4o to translate ALL text blocks in a single API call (Batching).
    Reduces API requests and avoids strict RPM limits.
    """
    logger.info("[Translator] Starting batch translation process.")
    parsed_data = state.get("parsed_json", {})
    memory = state.get("memory_dict", {})
    blocks = parsed_data.get("blocks", [])

    if not blocks:
        error_msg = "No blocks found to translate."
        logger.error(f"[Translator] {error_msg}")
        return {"translator_errors": error_msg}

    # Prepare lightweight batch input (only ID and Text to save tokens)
    batch_input = [{"id": b["id"], "text": b["text"]} for b in blocks]
    batch_input_str = json.dumps(batch_input, ensure_ascii=False, indent=2)

    llm = get_gpt4o_client()
    structured_llm = llm.with_structured_output(BatchTranslationResponse)

    # Load base system prompt from text file
    sys_prompt = load_prompt("translator.txt")

    # Inject dynamic memory dictionary if it exists
    if memory:
        sys_prompt += f"\n\nCRITICAL RULE:\nYou MUST strictly adhere to the following terminology dictionary: {json.dumps(memory, ensure_ascii=False)}"

    user_prompt = f"Text blocks to translate (JSON format):\n{batch_input_str}"

    # Retry Logic for HTTP 429 Rate Limits
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"[Translator] Sending {len(blocks)} blocks to GPT-4o (Attempt {attempt + 1}/{max_retries})...")

            response_data = structured_llm.invoke([
                SystemMessage(content=sys_prompt),
                HumanMessage(content=user_prompt)
            ])

            # Map translations by ID for O(1) lookup
            translations_dict = {item.id: item.target_text for item in response_data.translations}
            translated_blocks = []

            logger.info("\n" + "=" * 40 + "\n📝 BATCH TRANSLATION RESULTS\n" + "=" * 40)

            # Reassemble translated blocks with their original bounding boxes
            for block in blocks:
                block_id = block["id"]
                source_text = block["text"]
                # Fallback warning if the model skips an ID
                translated_text = translations_dict.get(block_id, "[Warning: Translation Missing]")

                translated_blocks.append({
                    "id": block_id,
                    "box": block["box"],
                    "source_text": source_text,
                    "target_text": translated_text
                })

                # Output translation pairs to logger
                logger.info(f"🆔 ID   : {block_id}")
                logger.info(f"🇬🇧 Src  : {source_text}")
                logger.info(f"🇨🇳 Trgt : {translated_text}")
                logger.info("-" * 40)

            logger.info("=" * 40)
            logger.info(f"[Translator] Successfully translated {len(translated_blocks)} blocks in ONE API call.")
            return {"translated_blocks": translated_blocks, "translator_errors": None}

        except Exception as e:
            error_msg = str(e).lower()
            logger.warning(f"[Translator] Batch translation failed on attempt {attempt + 1}: {e}")

            # Handle HTTP 429 Rate Limit specifically
            if "429" in error_msg or "rate limit" in error_msg:
                wait_time = 20
                logger.warning(f"[Translator] Rate limit reached. Waiting {wait_time}s before retrying...")
                time.sleep(wait_time)
            else:
                # Abort on other critical errors
                logger.error("[Translator] Critical translation error.", exc_info=True)
                return {"translator_errors": str(e)}

    final_err = "Failed to translate batch after maximum retries."
    logger.error(f"[Translator] {final_err}")
    return {"translator_errors": final_err}