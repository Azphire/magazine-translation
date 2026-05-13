import os
import base64
import json
from openai import OpenAI
from utils.logger import logger
from core.state import TranslationState

client = OpenAI()

def load_prompt(filename: str) -> str:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(base_dir, "prompts", filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def translator_node(state: TranslationState) -> dict:
    logger.info("[Translator] Starting multi-modal translation...")

    parsed_json = state.get("parsed_json", {})
    blocks = parsed_json.get("blocks", [])
    
    img_path = state.get("image_path")
    memory = state.get("memory_dict", {})

    if not blocks or not img_path:
        logger.warning("[Translator] No text blocks found from parser. Skipping translation.")
        return {"translated_blocks": []}

    base64_image = encode_image(img_path)
    source_texts = {str(i): b["text"] for i, b in enumerate(blocks)}

    # 1. Load External Prompt Template
    try:
        prompt_template = load_prompt("vision_translator.txt")
    except IOError:
        logger.error("[Translator] Prompt file not found.")
        return {"translated_blocks": []}

    # 2. Inject Dynamic Data
    prompt_text = prompt_template.replace(
        "[MEMORY]", json.dumps(memory, ensure_ascii=False)
    ).replace(
        "[SOURCE_TEXTS]", json.dumps(source_texts, ensure_ascii=False, indent=2)
    )

    try:
        # 3. Execute VLM Call with Contextual Image
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            temperature=0.3,
            max_tokens=4096 # Prevent JSON truncation for double-page spreads
        )

        translated_dict = json.loads(response.choices[0].message.content)
        translated_blocks = []

        logger.info("\n=== 🔍 [Translator] Output Review ===")
        for i, b in enumerate(blocks):
            original_text = b["text"]
            translated_text = translated_dict.get(str(i), original_text)

            preview_orig = (original_text[:40] + "...") if len(original_text) > 40 else original_text
            preview_trans = (translated_text[:40] + "...") if len(translated_text) > 40 else translated_text
            logger.info(f"Block [{i}]:")
            logger.info(f"  [EN] {preview_orig.replace(chr(10), ' ')}")
            logger.info(f"  [ZH] {preview_trans.replace(chr(10), ' ')}")

            translated_blocks.append({
                "source_text": original_text,
                "target_text": translated_text,
                "box": b["box"],
                "style": b.get("style", "body") # Pass through semantic style for Renderer
            })

        logger.info("====================================\n")
        logger.info(f"[Translator] Successfully translated {len(translated_blocks)} blocks.")

        return {"translated_blocks": translated_blocks}

    except Exception as e:
        logger.error(f"[Translator] API Call Failed: {e}")
        return {"translated_blocks": []}